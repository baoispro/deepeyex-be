package services

import (
	"auth-service/internal/config"
	"auth-service/internal/enums"
	"auth-service/internal/models"
	"auth-service/internal/repositories"
	"auth-service/internal/utils"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/google/uuid"
	"golang.org/x/crypto/bcrypt"
)

type AuthService struct {
	cfg       config.Config
	userRepo  *repositories.UserRepo
	tokenRepo *repositories.TokenRepo
}

func NewAuthService(cfg config.Config, ur *repositories.UserRepo, tr *repositories.TokenRepo) *AuthService {
	return &AuthService{cfg: cfg, userRepo: ur, tokenRepo: tr}
}

// ---------------- Register ----------------
func (s *AuthService) Register(username, password string, role enums.Role) error {
	if role != enums.Patient && role != enums.Doctor && role != enums.Admin {
		return errors.New("invalid role")
	}
	if _, err := s.userRepo.FindByUsername(username); err == nil {
		return errors.New("username already exists")
	}

	hashedPwd, _ := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
	u := &models.User{
		ID:       uuid.NewString(),
		Username: username,
		Password: string(hashedPwd),
		Role:     role,
	}
	return s.userRepo.Create(u)
}

// ---------------- Login ----------------
func (s *AuthService) Login(username, password string) (access string, aExp time.Time, refresh string, rExp time.Time, u *models.User, err error) {
    u, err = s.userRepo.FindByUsername(username)
    if err != nil {
        return "", time.Time{}, "", time.Time{}, nil, errors.New("invalid credentials")
    }
    if bcrypt.CompareHashAndPassword([]byte(u.Password), []byte(password)) != nil {
        return "", time.Time{}, "", time.Time{}, nil, errors.New("invalid credentials")
    }

    // Lấy key/secret từ config hoặc DB (do bạn add trong Kong)
    kongKey := "tMK7AdrE6lHxsFU1Qk0EPczXYvRKLw4K"
    kongSecret := "Cdes9ghOllkzh0ZgmOvLdH97tdjImE42"

    // Sinh access token theo Kong yêu cầu
    aExp = time.Now().Add(time.Hour)
    claims := jwt.MapClaims{
        "iss": kongKey,
        "exp": aExp.Unix(),
        "sub": u.ID,
        "role": string(u.Role),
    }
    token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
    access, err = token.SignedString([]byte(kongSecret))
    if err != nil {
        return "", time.Time{}, "", time.Time{}, nil, err
    }

    // Refresh token logic: bạn vẫn có thể giữ như cũ (DB quản lý)
    refresh, rExp, _ = utils.GenerateRefreshToken(s.cfg, u.ID)
    if err := s.tokenRepo.Save(u.ID, hashToken(refresh), rExp); err != nil {
        return "", time.Time{}, "", time.Time{}, nil, err
    }

    return access, aExp, refresh, rExp, u, nil
}

// ---------------- Refresh ----------------
func (s *AuthService) Refresh(oldRefresh string) (access string, aExp time.Time, newRefresh string, newExp time.Time, err error) {
	claims, err := utils.ParseRefreshToken(s.cfg, oldRefresh)
	if err != nil || claims == nil {
		return "", time.Time{}, "", time.Time{}, errors.New("invalid refresh token")
	}
	userID := claims.Subject

	ok, uid, err := s.tokenRepo.IsValid(hashToken(oldRefresh))
	if err != nil || !ok || uid != userID {
		return "", time.Time{}, "", time.Time{}, errors.New("refresh token invalid/revoked")
	}

	u, err := s.userRepo.FindByID(userID)
	if err != nil {
		return "", time.Time{}, "", time.Time{}, errors.New("user not found")
	}

	// Generate new access token
	access, aExp, _ = utils.GenerateAccessToken(s.cfg, u.ID, string(u.Role))

	// Revoke old refresh token
	if err := s.tokenRepo.RevokeByHash(hashToken(oldRefresh)); err != nil {
		return "", time.Time{}, "", time.Time{}, err
	}

	// Generate and save new refresh token
	newRefresh, newExp, _ = utils.GenerateRefreshToken(s.cfg, u.ID)
	if err := s.tokenRepo.Save(u.ID, hashToken(newRefresh), newExp); err != nil {
		return "", time.Time{}, "", time.Time{}, err
	}

	return access, aExp, newRefresh, newExp, nil
}

// ---------------- Logout ----------------
func (s *AuthService) Logout(refresh string) {
	if refresh != "" {
		_ = s.tokenRepo.RevokeByHash(hashToken(refresh))
	}
}

// ---------------- Helpers ----------------
func hashToken(token string) string {
	h := sha256.Sum256([]byte(token))
	return hex.EncodeToString(h[:])
}

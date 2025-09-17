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
	"regexp"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/google/uuid"
	"golang.org/x/crypto/bcrypt"
	"gorm.io/gorm"
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
func (s *AuthService) Register(username, email, password, firebaseUID string) error {
	if _, err := s.userRepo.FindByUsername(username); err == nil {
		return errors.New("username already exists")
	}

	if _, err := s.userRepo.FindByEmail(email); err == nil {
		return errors.New("email already exists")
	}

	if len([]rune(password)) < 8 {
		return errors.New("password must be at least 8 characters long")
	}

	if !isValidPassword(password) {
		return errors.New("password must contain at least 1 uppercase, 1 lowercase, 1 digit, and 1 special character")
	}

	hashedPwd, _ := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
	u := &models.User{
		ID:          uuid.NewString(),
		Username:    username,
		Password:    string(hashedPwd),
		Role:        enums.Patient,
		Email:       email,
		FirebaseUID: firebaseUID,
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

	access, aExp, refresh, rExp, err = s.generateTokensKong(u.ID, string(u.Role))
	if err != nil {
		return "", time.Time{}, "", time.Time{}, nil, err
	}
	return access, aExp, refresh, rExp, u, nil
}

// ---------------- Login Firebase ----------------
func (s *AuthService) LoginFirebase(firebaseUID, email string) (access string, aExp time.Time, refresh string, rExp time.Time, u *models.User, err error) {
	u, err = s.userRepo.FindByFirebaseUID(firebaseUID)
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			u = &models.User{
				ID:          uuid.NewString(),
				FirebaseUID: firebaseUID,
				Email:       email,
				Role:        enums.Patient,
				CreatedAt:   time.Now(),
				UpdatedAt:   time.Now(),
			}
			if err := s.userRepo.Create(u); err != nil {
				return "", time.Time{}, "", time.Time{}, nil, err
			}
		} else {
			return "", time.Time{}, "", time.Time{}, nil, err
		}
	}

	access, aExp, refresh, rExp, err = s.generateTokensKong(u.ID, string(u.Role))
	if err != nil {
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

	// Generate new access token theo Kong
	access, aExp, err = s.generateAccessTokenKong(u.ID, string(u.Role))
	if err != nil {
		return "", time.Time{}, "", time.Time{}, err
	}

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
func (s *AuthService) generateTokensKong(userID string, role string) (access string, aExp time.Time, refresh string, rExp time.Time, err error) {
	// Access token ký theo Kong
	access, aExp, err = s.generateAccessTokenKong(userID, role)
	if err != nil {
		return "", time.Time{}, "", time.Time{}, err
	}

	// Refresh token vẫn tự quản lý
	refresh, rExp, err = utils.GenerateRefreshToken(s.cfg, userID)
	if err != nil {
		return "", time.Time{}, "", time.Time{}, err
	}
	if err := s.tokenRepo.Save(userID, hashToken(refresh), rExp); err != nil {
		return "", time.Time{}, "", time.Time{}, err
	}
	return access, aExp, refresh, rExp, nil
}

func (s *AuthService) generateAccessTokenKong(userID string, role string) (string, time.Time, error) {
	kongKey := s.cfg.KongKey // load từ config.yml hoặc env
	kongSecret := s.cfg.KongSecret

	aExp := time.Now().Add(time.Hour)
	claims := jwt.MapClaims{
		"iss":  kongKey,
		"sub":  userID,
		"role": role,
		"exp":  aExp.Unix(),
	}
	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	signed, err := token.SignedString([]byte(kongSecret))
	return signed, aExp, err
}

func hashToken(token string) string {
	h := sha256.Sum256([]byte(token))
	return hex.EncodeToString(h[:])
}

func isValidPassword(pwd string) bool {
	var (
		hasUpper   = regexp.MustCompile(`[A-Z]`)
		hasLower   = regexp.MustCompile(`[a-z]`)
		hasNumber  = regexp.MustCompile(`[0-9]`)
		hasSpecial = regexp.MustCompile(`[^A-Za-z0-9]`)
	)

	return hasUpper.MatchString(pwd) &&
		hasLower.MatchString(pwd) &&
		hasNumber.MatchString(pwd) &&
		hasSpecial.MatchString(pwd)
}

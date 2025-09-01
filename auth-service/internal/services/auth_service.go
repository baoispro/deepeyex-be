package services

import (
	"auth-service/internal/config"
	"auth-service/internal/enums"
	"auth-service/internal/models"
	"auth-service/internal/repositories"
	"auth-service/internal/utils"
	"errors"
	"time"

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

// Register user
func (s *AuthService) Register(username, password string, role enums.Role) error {
	if role != enums.Patient && role != enums.Doctor && role != enums.Admin {
		return errors.New("invalid role")
	}
	if _, err := s.userRepo.FindByUsername(username); err == nil {
		return errors.New("username already exists")
	}
	pwd, _ := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
	u := &models.User{
		ID:       uuid.NewString(),
		Username: username,
		Password: string(pwd),
		Role:     role,
	}
	return s.userRepo.Create(u)
}

// Login -> return access, refresh, expires, user info
func (s *AuthService) Login(username, password string) (access string, aExp time.Time, refresh string, rExp time.Time, u *models.User, err error) {
	u, err = s.userRepo.FindByUsername(username)
	if err != nil {
		return "", time.Time{}, "", time.Time{}, nil, errors.New("invalid credentials")
	}
	if bcrypt.CompareHashAndPassword([]byte(u.Password), []byte(password)) != nil {
		return "", time.Time{}, "", time.Time{}, nil, errors.New("invalid credentials")
	}

	access, aExp, _ = utils.GenerateAccessToken(s.cfg, u.ID, string(u.Role))
	refresh, rExp, _ = utils.GenerateRefreshToken(s.cfg, u.ID)

	// Save refresh token hash
	if err := s.tokenRepo.Save(u.ID, hashString(refresh), rExp); err != nil {
		return "", time.Time{}, "", time.Time{}, nil, err
	}
	return access, aExp, refresh, rExp, u, nil
}

// Refresh token -> new access, new refresh
func (s *AuthService) Refresh(oldRefresh string) (access string, aExp time.Time, newRefresh string, newExp time.Time, err error) {
	userIDClaims, err := utils.ParseRefreshToken(s.cfg, oldRefresh)
	if err != nil {
		return "", time.Time{}, "", time.Time{}, err
	}
	var userID string
	if userIDClaims != nil {
		userID = userIDClaims.Subject
	} else {
		return "", time.Time{}, "", time.Time{}, errors.New("invalid refresh token claims")
	}

	ok, uid, err := s.tokenRepo.IsValid(hashString(oldRefresh))
	if err != nil || !ok || uid != userID {
		return "", time.Time{}, "", time.Time{}, errors.New("refresh token invalid/revoked")
	}

	u, err := s.userRepo.FindByID(userID)
	if err != nil {
		return "", time.Time{}, "", time.Time{}, errors.New("user not found")
	}

	// cấp access mới
	access, aExp, _ = utils.GenerateAccessToken(s.cfg, u.ID, string(u.Role))

	// rotate refresh token
	_ = s.tokenRepo.RevokeByHash(hashString(oldRefresh))
	newRefresh, newExp, _ = utils.GenerateRefreshToken(s.cfg, u.ID)
	_ = s.tokenRepo.Save(u.ID, hashString(newRefresh), newExp)

	return access, aExp, newRefresh, newExp, nil
}

// Logout -> revoke token
func (s *AuthService) Logout(refresh string) {
	if refresh != "" {
		_ = s.tokenRepo.RevokeByHash(hashString(refresh))
	}
}

// ----- helpers -----
func hashString(s string) string {
	b, _ := bcrypt.GenerateFromPassword([]byte(s), bcrypt.DefaultCost)
	return string(b)
}

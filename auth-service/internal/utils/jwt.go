package utils

import (
	"auth-service/internal/config"
	"errors"
	"time"

	"github.com/golang-jwt/jwt/v5"
)

type AccessClaims struct {
	UserID string `json:"uid"`
	Role   string `json:"role"`
	jwt.RegisteredClaims
}

// GenerateRefreshToken sinh refresh token kèm thời gian hết hạn
func GenerateRefreshToken(cfg config.Config, userID string) (string, time.Time, error) {
	exp := time.Now().Add(time.Duration(cfg.RefreshTTLDays) * 24 * time.Hour)
	claims := jwt.RegisteredClaims{
		Subject:   userID,
		ExpiresAt: jwt.NewNumericDate(exp),
		IssuedAt:  jwt.NewNumericDate(time.Now()),
	}
	t := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	s, err := t.SignedString([]byte(cfg.RefreshSecret))
	return s, exp, err
}

// ParseRefreshToken parse + validate refresh token
func ParseRefreshToken(cfg config.Config, tokenStr string) (*jwt.RegisteredClaims, error) {
	token, err := jwt.ParseWithClaims(tokenStr, &jwt.RegisteredClaims{}, func(token *jwt.Token) (interface{}, error) {
		if _, ok := token.Method.(*jwt.SigningMethodHMAC); !ok {
			return nil, errors.New("unexpected signing method")
		}
		return []byte(cfg.RefreshSecret), nil
	})

	if err != nil {
		return nil, err
	}

	if claims, ok := token.Claims.(*jwt.RegisteredClaims); ok && token.Valid {
		return claims, nil
	}
	return nil, errors.New("invalid refresh token")
}

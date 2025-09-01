package repositories

import (
	"auth-service/internal/models"
	"time"

	"gorm.io/gorm"
)

type TokenRepo struct{ db *gorm.DB }

func NewTokenRepo(db *gorm.DB) *TokenRepo { return &TokenRepo{db: db} }

func (r *TokenRepo) Save(userID, tokenHash string, expires time.Time) error {
	if err := r.db.Create(&models.RefreshToken{
		UserID:    userID,
		TokenHash: tokenHash,
		ExpiresAt: expires,
	}).Error; err != nil {
		return err
	}
	return nil
}

func (r *TokenRepo) RevokeByHash(tokenHash string) error {
	return r.db.Model(&models.RefreshToken{}).
		Where("token_hash = ? AND revoked = FALSE", tokenHash).
		Update("revoked", true).Error
}

func (r *TokenRepo) IsValid(tokenHash string) (bool, string, error) {
	var t models.RefreshToken
	if err := r.db.
		Where("token_hash = ? AND revoked = FALSE AND expires_at > NOW()", tokenHash).
		First(&t).Error; err != nil {
		if err == gorm.ErrRecordNotFound {
			return false, "", nil
		}
		return false, "", err
	}
	return true, t.UserID, nil
}

func (r *TokenRepo) RevokeAllForUser(userID string) error {
	return r.db.Model(&models.RefreshToken{}).
		Where("user_id = ? AND revoked = FALSE", userID).
		Update("revoked", true).Error
}

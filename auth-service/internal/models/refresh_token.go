package models

import "time"

type RefreshToken struct {
	ID        uint      `gorm:"primaryKey" json:"-"`
	UserID    string    `gorm:"index;not null" json:"-"`
	TokenHash string    `gorm:"uniqueIndex;not null" json:"-"`
	ExpiresAt time.Time `json:"-"`
	Revoked   bool      `gorm:"default:false" json:"-"`
	CreatedAt time.Time `json:"-"`
	UpdatedAt time.Time `json:"-"`
}

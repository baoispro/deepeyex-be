package models

import "time"

type User struct {
	ID        uint   `json:"id" gorm:"primaryKey"`
	Email     string `gorm:"unique;not null" json:"email"`
	Password  string `gorm:"not null" json:"-"`
	CreatedAt time.Time
	UpdatedAt time.Time
}

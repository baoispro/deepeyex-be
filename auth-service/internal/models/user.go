package models

import (
	"auth-service/internal/enums"
	"time"
)

type User struct {
	ID        string     `json:"id" gorm:"primaryKey"`
	Username  string     `json:"username" gorm:"uniqueIndex;size:100;not null"`
	Password  string     `json:"password"`
	Role      enums.Role `json:"role" gorm:"size:20;not null"` // patient/doctor/admin
	CreatedAt time.Time  `json:"created_at"`
	UpdatedAt time.Time  `json:"updated_at"`
}

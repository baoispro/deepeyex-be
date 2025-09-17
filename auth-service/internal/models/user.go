package models

import (
	"auth-service/internal/enums"
	"time"
)

type User struct {
	ID          string     `json:"id" gorm:"primaryKey"`
	Username    string     `json:"username" gorm:"uniqueIndex;size:100"`
	Email       string     `json:"email" gorm:"uniqueIndex;size:255"`
	Password    string     `json:"password"`
	FirebaseUID string     `json:"firebase_uid" gorm:"uniqueIndex;size:128"`
	Role        enums.Role `json:"role" gorm:"size:20;not null"` // patient/doctor/admin
	CreatedAt   time.Time  `json:"created_at"`
	UpdatedAt   time.Time  `json:"updated_at"`
}

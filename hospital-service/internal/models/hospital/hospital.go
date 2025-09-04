package hospital

import (
	"time"
)

type Hospital struct {
	HospitalID string    `json:"hospital_id" gorm:"type:uuid;primaryKey"`
	Name       string    `json:"name" gorm:"size:100;not null"`
	Address    string    `json:"address" gorm:"size:255"`
	Phone      string    `json:"phone" gorm:"size:20"`
	Email      string    `json:"email" gorm:"size:100"`
	LogoURL    string    `json:"logo_url" gorm:"size:255"`
	CreatedAt  time.Time `json:"created_at" gorm:"autoCreateTime"`
	UpdatedAt  time.Time `json:"updated_at" gorm:"autoUpdateTime"`
}

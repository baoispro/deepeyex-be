package hospital

import (
	"hospital-service/internal/models/doctor"
	"time"
)

type Hospital struct {
	HospitalID string          `json:"hospital_id" gorm:"type:uuid;primaryKey"`
	Name       string          `json:"name" gorm:"size:100;not null"`
	Address    string          `json:"address" gorm:"size:255"`
	Phone      string          `json:"phone" gorm:"size:20"`
	Email      string          `json:"email" gorm:"size:100"`
	Image      string          `json:"image" gorm:"size:255"`
	CreatedAt  time.Time       `json:"created_at" gorm:"autoCreateTime"`
	UpdatedAt  time.Time       `json:"updated_at" gorm:"autoUpdateTime"`
	Doctors    []doctor.Doctor `gorm:"foreignKey:HospitalID;constraint:OnUpdate:CASCADE,OnDelete:SET NULL;"`
	Slug       string          `json:"slug" gorm:"size:150;uniqueIndex"`
	UrlMap     string          `json:"url_map"`
}

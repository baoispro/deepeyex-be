package doctor

import (
	"hospital-service/internal/enums"
	"hospital-service/internal/models/hospital"
	"time"
)

type Doctor struct {
	DoctorID   string            `json:"doctor_id" gorm:"type:uuid;primaryKey"`
	UserID     string            `json:"user_id" gorm:"type:uuid;not null"`
	FullName   string            `json:"full_name" gorm:"size:100;not null"`
	Phone      string            `json:"phone" gorm:"size:20"`
	Email      string            `json:"email" gorm:"size:100"`
	AvatarURL  string            `json:"avatar_url" gorm:"size:255"`
	Specialty  enums.Specialty   `json:"specialty" gorm:"size:50;not null"`
	HospitalID string            `json:"hospital_id" gorm:"type:uuid;not null"`
	Hospital   hospital.Hospital `gorm:"foreignKey:HospitalID;references:HospitalID;constraint:OnUpdate:CASCADE,OnDelete:CASCADE"`
	CreatedAt  time.Time         `json:"created_at" gorm:"autoCreateTime"`
	UpdatedAt  time.Time         `json:"updated_at" gorm:"autoUpdateTime"`
}

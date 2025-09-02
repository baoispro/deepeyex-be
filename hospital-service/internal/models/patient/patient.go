package patient

import (
	"hospital-service/internal/enums"
	"time"
)

type Patient struct {
	PatientID string       `json:"patient_id" gorm:"primaryKey"`
	UserID    string       `json:"user_id" gorm:"size:36;not null"`
	FullName  string       `json:"full_name" gorm:"size:100;not null"`
	DOB       time.Time    `json:"dob"`
	Gender    enums.Gender `json:"gender" gorm:"size:10;not null"`
	Address   string       `json:"address" gorm:"size:255"`
	Phone     string       `json:"phone" gorm:"size:20"`
	Email     string       `json:"email" gorm:"size:100"`

	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
}

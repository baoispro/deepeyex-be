package doctor

import (
	"hospital-service/internal/enums"
	"time"
)

type Doctor struct {
    DoctorID  string `json:"doctor_id" gorm:"primaryKey"`
    UserID    string `json:"user_id" gorm:"size:36;not null"`
	FullName  string `json:"full_name" gorm:"size:100;not null"`
    Phone     string `json:"phone" gorm:"size:20"`
	Email     string `json:"email" gorm:"size:100"`
 	Specialty enums.Specialty `json:"specialty" gorm:"size:50;not null"`
	HospitalID string               `json:"hospital_id" gorm:"type:uuid;not null"` // FK
	// Hospital   hospital.Hospital    `json:"hospital" gorm:"foreignKey:HospitalID;references:HospitalID"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
}

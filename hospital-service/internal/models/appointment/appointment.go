package appointment

import (
	"hospital-service/internal/enums"
	"time"
)

type Appointment struct {
	AppointmentID string            `json:"appointment_id" gorm:"type:uuid;primaryKey"`
	PatientID     string            `json:"patient_id" gorm:"type:uuid;not null"`
	DoctorID      string            `json:"doctor_id" gorm:"type:uuid;not null"`
	HospitalID    string            `json:"hospital_id" gorm:"type:uuid;not null"`
	ScheduledTime time.Time         `json:"scheduled_time" gorm:"not null"`
	Status        enums.AppointmentStatus `json:"status" gorm:"type:varchar(20);default:'PENDING'"`
	Notes         string            `json:"notes" gorm:"size:255"`

	CreatedAt time.Time `json:"created_at" gorm:"autoCreateTime"`
	UpdatedAt time.Time `json:"updated_at" gorm:"autoUpdateTime"`
}
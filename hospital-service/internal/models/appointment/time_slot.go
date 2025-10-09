package appointment

import (
	"hospital-service/internal/models/doctor"
	"time"
)

type TimeSlot struct {
	SlotID        string       `gorm:"column:slot_id;primaryKey" json:"slot_id"`
	DoctorID      string       `gorm:"type:uuid;not null" json:"doctor_id"`
	StartTime     time.Time    `gorm:"not null" json:"start_time"`
	EndTime       time.Time    `gorm:"not null" json:"end_time"`
	Capacity      int          `gorm:"not null;default:1" json:"capacity"`
	CreatedAt     time.Time    `gorm:"autoCreateTime" json:"created_at"`
	UpdatedAt     time.Time    `gorm:"autoUpdateTime" json:"updated_at"`
	AppointmentID *string      `gorm:"index" json:"appointment_id"`
    Appointment   *Appointment `gorm:"constraint:OnUpdate:CASCADE,OnDelete:SET NULL" json:"appointment,omitempty"`

	Doctor *doctor.Doctor `gorm:"foreignKey:DoctorID;references:DoctorID" json:"doctor,omitempty"`
}

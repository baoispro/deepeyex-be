package appointment

import (
	"time"

	"hospital-service/internal/enums"
	"hospital-service/internal/models/doctor"
	"hospital-service/internal/models/hospital"
	"hospital-service/internal/models/patient"
)

// Appointment model đại diện cho lịch hẹn khám bệnh
type Appointment struct {
	AppointmentID   string                  `gorm:"column:appointment_id;primaryKey;size:36" json:"appointment_id"`
	AppointmentCode string                  `gorm:"size:64;unique;not null" json:"appointment_code"` // Mã lịch hẹn duy nhất
	PatientID       string                  `gorm:"not null;size:36" json:"patient_id"`              // ID bệnh nhân
	HospitalID      string                  `gorm:"not null;size:36" json:"hospital_id"`             // ID bệnh viện
	DoctorID        string                  `gorm:"not null;size:36" json:"doctor_id"`               // ID bác sĩ
	Status          enums.AppointmentStatus `gorm:"type:appointment_status;default:'PENDING'" json:"status"`
	Notes           *string                 `gorm:"type:text" json:"notes,omitempty"` // Ghi chú thêm từ bệnh nhân
	CreatedAt       time.Time               `gorm:"autoCreateTime" json:"created_at"`
	UpdatedAt       time.Time               `gorm:"autoUpdateTime" json:"updated_at"`
	CheckedInAt     *time.Time              `json:"checked_in_at,omitempty"` // Thời gian bệnh nhân check-in (nếu có)
	BookUserId      string                  `gorm:"not null;size:36" json:"book_user_id"`

	// Quan hệ
	TimeSlots []TimeSlot        `gorm:"foreignKey:AppointmentID;references:AppointmentID" json:"time_slots,omitempty"`
	Patient  patient.Patient   `gorm:"foreignKey:PatientID;references:PatientID" json:"patient"`
	Hospital hospital.Hospital `gorm:"foreignKey:HospitalID;references:HospitalID" json:"hospital,omitempty"`
	Doctor   doctor.Doctor     `gorm:"foreignKey:DoctorID;references:DoctorID" json:"doctor,omitempty"`
}

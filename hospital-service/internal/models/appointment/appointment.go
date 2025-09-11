package appointment

import (
	"time"

	"hospital-service/internal/enums"
	"hospital-service/internal/models/patient"
)

// Appointment model đại diện cho lịch hẹn khám bệnh
type Appointment struct {
	AppointmentID   string                 `gorm:"column:appointment_id;primaryKey;size:36" json:"appointment_id"`
	AppointmentCode string                 `gorm:"size:64;unique;not null" json:"appointment_code"` // Mã lịch hẹn duy nhất
	SlotID          string                 `gorm:"not null;size:36" json:"slot_id"`                // ID của khung giờ đã chọn
	PatientID       string                 `gorm:"not null;size:36" json:"patient_id"`            // ID bệnh nhân
	HospitalID      string                 `gorm:"not null;size:36" json:"hospital_id"`           // ID bệnh viện
	Specialty       enums.Specialty        `gorm:"type:varchar(50);not null" json:"specialty"`    // Chuyên khoa
	DoctorID        string                 `gorm:"not null;size:36" json:"doctor_id"`             // ID bác sĩ
	Status 			enums.AppointmentStatus `gorm:"type:appointment_status;default:'PENDING'" json:"status"`
	Notes           *string                `gorm:"type:text" json:"notes,omitempty"`             // Ghi chú thêm từ bệnh nhân
	CreatedAt       time.Time              `gorm:"autoCreateTime" json:"created_at"`
	UpdatedAt       time.Time              `gorm:"autoUpdateTime" json:"updated_at"`
	CheckedInAt     *time.Time             `json:"checked_in_at,omitempty"` // Thời gian bệnh nhân check-in (nếu có)

	// Quan hệ
	 TimeSlot TimeSlot 						`gorm:"foreignKey:SlotID;references:SlotID" json:"timeSlot"`
    Patient  patient.Patient  				`gorm:"foreignKey:PatientID;references:PatientID" json:"patient"`
}

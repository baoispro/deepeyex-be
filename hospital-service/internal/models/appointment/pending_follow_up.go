package appointment

import (
	"time"
)

// PendingFollowUpAppointment model đại diện cho lịch tái khám chờ xác nhận từ bệnh nhân
type PendingFollowUpAppointment struct {
	PendingID       string     `gorm:"column:pending_id;primaryKey;size:36" json:"pending_id"`
	PatientID       string     `gorm:"not null;size:36" json:"patient_id"`        // ID bệnh nhân
	HospitalID      string     `gorm:"not null;size:36" json:"hospital_id"`       // ID bệnh viện
	DoctorID        string     `gorm:"not null;size:36" json:"doctor_id"`         // ID bác sĩ
	SlotIDs         string     `gorm:"type:text" json:"slot_ids"`                  // Danh sách slot IDs (JSON array string)
	ServiceName     string     `gorm:"size:255" json:"service_name"`               // Tên dịch vụ
	RelatedRecordID *string    `gorm:"size:36" json:"related_record_id,omitempty"` // ID medical record liên quan (optional)
	ConfirmationToken string   `gorm:"size:64;unique" json:"confirmation_token"`    // Token xác nhận duy nhất
	Status          string     `gorm:"size:20;default:'PENDING'" json:"status"`   // PENDING, CONFIRMED, EXPIRED
	Notes           string     `gorm:"type:text" json:"notes,omitempty"`          // Ghi chú
	ExpiresAt       time.Time  `gorm:"not null" json:"expires_at"`                  // Thời gian hết hạn (7 ngày)
	CreatedAt       time.Time  `gorm:"autoCreateTime" json:"created_at"`
	UpdatedAt       time.Time  `gorm:"autoUpdateTime" json:"updated_at"`
	ConfirmedAt     *time.Time `json:"confirmed_at,omitempty"`                    // Thời gian xác nhận

	// Thông tin thêm để hiển thị
	DoctorName  string `json:"doctor_name,omitempty" gorm:"-"`  // Tên bác sĩ
	PatientName string `json:"patient_name,omitempty" gorm:"-"` // Tên bệnh nhân
	HospitalName string `json:"hospital_name,omitempty" gorm:"-"` // Tên bệnh viện
}


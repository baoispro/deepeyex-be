package medicalrecord

import (
	"hospital-service/internal/enums"
	"time"
)

type MedicationReminder struct {
	ID                 string               `gorm:"primaryKey;size:36" json:"id"`
	PrescriptionItemID string               `gorm:"size:36;not null" json:"prescription_item_id"`
	ReminderTime       time.Time            `json:"reminder_time"`
	Status             enums.ReminderStatus `gorm:"size:20;default:'PENDING'" json:"status"` // PENDING, DONE, SKIPPED
	CreatedAt          time.Time            `gorm:"autoCreateTime" json:"created_at"`
}

// MedicationReminderWithItem chứa thông tin reminder kèm thông tin thuốc
type MedicationReminderWithItem struct {
	ID                 string    `json:"id"`
	PrescriptionItemID string    `json:"prescription_item_id"`
	ReminderTime       time.Time `json:"reminder_time"`
	Status             string    `json:"status"`
	CreatedAt          time.Time `json:"created_at"`
	// PrescriptionItem info
	DrugName           string    `json:"drug_name"`
	Dosage             string    `json:"dosage"`
	Frequency          string    `json:"frequency"`
	Notes              *string   `json:"notes,omitempty"`
}

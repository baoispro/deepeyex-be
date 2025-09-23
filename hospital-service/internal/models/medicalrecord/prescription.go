package medicalrecord

import "time"

type Prescription struct {
	PrescriptionID string       `gorm:"primaryKey;size:36" json:"prescription_id"`
	RecordID       string       `gorm:"size:36;not null" json:"record_id"`
	Status         string       `gorm:"size:20;default:'PENDING'" json:"status"` // PENDING, APPROVED, REJECTED
	ApprovedBy     string       `gorm:"size:36" json:"approved_by,omitempty"`
	ApprovedAt     *time.Time   `json:"approved_at,omitempty"` 
	Items          []PrescriptionItem `gorm:"foreignKey:PrescriptionID;references:PrescriptionID" json:"items,omitempty"`
	CreatedAt      time.Time `gorm:"autoCreateTime"`
	UpdatedAt      time.Time `gorm:"autoUpdateTime"`
}

type PrescriptionItem struct {
	ItemID        string `gorm:"primaryKey;size:36" json:"prescription_item_id"`
	PrescriptionID string `gorm:"size:36;not null" json:"prescription_id"`
	DrugName      string `gorm:"size:100" json:"drug_name"`
	Dosage        string `gorm:"size:50" json:"dosage"`
	Frequency     string `gorm:"size:50" json:"frequency"`
	DurationDays  int 	`gorm:"not null" json:"duration_days"`
}

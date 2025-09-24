package medicalrecord

import "time"

type Attachment struct {
	ID        string    `gorm:"primaryKey;size:36" json:"id"`
	RecordID  string    `gorm:"size:36;not null" json:"record_id"`
	FileURL   string    `gorm:"size:255" json:"file_url"`
	FileType  string    `gorm:"size:20" json:"file_type"`
	CreatedAt time.Time `gorm:"autoCreateTime" json:"created_at"`
}
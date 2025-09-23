package medicalrecord

import "time"

type FollowUp struct {
	FollowUpID  string    `gorm:"primaryKey;size:36" json:"follow_up_id"`
	RecordID    string    `gorm:"size:36;not null" json:"record_id"`
	NextAppointment *time.Time `gorm:"not null" json:"next_appointment"`
	Notes       string   `gorm:"type:text" json:"notes"`
	CreatedAt   time.Time `gorm:"autoCreateTime" json:"created_at"` 
}

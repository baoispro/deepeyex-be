package notification

import "time"

type Notification struct {
	ID        string    `gorm:"primaryKey;size:36" json:"id"`
	UserID    string    `gorm:"size:36;index" json:"userId"`       // liên kết đến user nhận thông báo
	Title     string    `gorm:"size:255;not null" json:"title"`
	Message   string    `gorm:"type:text" json:"message"`
	// Type      NotificationType `gorm:"size:20;default:SYSTEM"`
	TargetURL string    `gorm:"size:255" json:"targetUrl"`         // đường dẫn chuyển đến khi click
	Read      bool      `gorm:"default:false" json:"read"`
	CreatedAt time.Time `gorm:"autoCreateTime" json:"createdAt"`
	UpdatedAt time.Time `gorm:"autoUpdateTime" json:"updatedAt"`
}

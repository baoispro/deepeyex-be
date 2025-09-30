package service

import "time"

// Service (bảng cha)
type Service struct {
    ServiceID string    `json:"service_id" gorm:"primaryKey;size:36"`
    Name      string    `json:"name" gorm:"size:100;not null"`
    Duration  int       `json:"duration" gorm:"not null"`
    Price     float64   `json:"price" gorm:"not null"`
    CreatedAt time.Time `json:"created_at"`
    UpdatedAt time.Time `json:"updated_at"`
}

func (Service) TableName() string {
    return "services"
}

// DoctorService (bảng con)
type DoctorService struct {
    DoctorID  string    `json:"doctor_id" gorm:"size:36;not null;index"`
    ServiceID string    `json:"service_id" gorm:"size:36;not null;index"`

    // Quan hệ chuẩn: DoctorService -> Service
    Service   Service   `gorm:"foreignKey:ServiceID;references:ServiceID" json:"service"`

    CreatedAt time.Time `json:"created_at"`
    UpdatedAt time.Time `json:"updated_at"`
}

func (DoctorService) TableName() string {
    return "doctor_services"
}

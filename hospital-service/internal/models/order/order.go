package order

import (
	"hospital-service/internal/enums"
	"hospital-service/internal/models/appointment"
	"hospital-service/internal/models/patient"
	"time"
)

type Order struct {
	OrderID       string            `gorm:"column:order_id;primaryKey;size:36" json:"order_id"`
	PatientID     string            `gorm:"not null" json:"patient_id"`
	AppointmentID string            `gorm:"column:appointment_id;size:36" json:"appointment_id"`
	BookUserId    string            `gorm:"not null;size:36" json:"book_user_id"`
	CreatedAt     time.Time         `gorm:"autoCreateTime" json:"created_at"`
	Status        enums.OrderStatus `gorm:"type:order_status;default:'PENDING'" json:"status"`
	TotalAmount   float64           `gorm:"type:decimal(10,2)" json:"total_amount"`
	OrderItems    []OrderItem       `gorm:"foreignKey:OrderID;constraint:OnDelete:CASCADE" json:"order_items"`
	Patient       patient.Patient   `gorm:"foreignKey:PatientID;references:PatientID" json:"patient"`
	Appointment appointment.Appointment `gorm:"foreignKey:AppointmentID;references:AppointmentID" json:"appointment,omitempty"`
}

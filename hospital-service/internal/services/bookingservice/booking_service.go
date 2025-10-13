package bookingservice

import (
	"errors"
	"fmt"
	"hospital-service/internal/enums"
	"hospital-service/internal/models/appointment"
	"hospital-service/internal/models/order"
	"hospital-service/internal/services/appointmentservice"
	"hospital-service/internal/services/orderservice"
)

// Request struct để tạo booking
type BookingRequest struct {
	PatientID     string                          `json:"patient_id" binding:"required"`
	DoctorID      string                          `json:"doctor_id" binding:"required"`
	HospitalID    string                          `json:"hospital_id" binding:"required"`
	SlotIDs       []string                        `json:"slot_ids" binding:"required"` // đổi từ SlotID sang SlotIDs
	BookUserID    string                          `json:"book_user_id" binding:"required"`
	ServiceName   string                          `json:"service_name" binding:"required"`
	Notes         string                          `json:"notes,omitempty"`
	OrderItems    []orderservice.OrderItemRequest `json:"order_items" binding:"required"`
	PaymentStatus enums.OrderStatus               `json:"payment_status" binding:"required"` // thêm field này
}

// Response struct
type BookingResponse struct {
	Appointment *appointment.Appointment `json:"appointment"`
	Order       *order.Order             `json:"order"`
}

type BookingService struct {
	appointmentService *appointmentservice.AppointmentService
	orderService       *orderservice.OrderService
}

func NewBookingService(apptSvc *appointmentservice.AppointmentService, ordSvc *orderservice.OrderService) *BookingService {
	return &BookingService{
		appointmentService: apptSvc,
		orderService:       ordSvc,
	}
}

func (s *BookingService) CreateBooking(req BookingRequest) (*BookingResponse, error) {
	if req.PatientID == "" || req.DoctorID == "" || req.HospitalID == "" || len(req.SlotIDs) == 0 || req.BookUserID == "" {
		return nil, errors.New("missing required fields")
	}

	// 1. Tạo appointment
	appt, err := s.appointmentService.Create(req.PatientID, req.DoctorID, req.HospitalID, req.BookUserID, req.SlotIDs, req.Notes, req.ServiceName)
	if err != nil {
		return nil, fmt.Errorf("failed to create appointment: %v", err)
	}

	// 2. Tạo order gắn với appointment
	ord, err := s.orderService.CreateOrder(req.PatientID, appt.AppointmentID, req.BookUserID, req.PaymentStatus, req.OrderItems, nil)
	if err != nil {
		// Nếu tạo order fail, rollback appointment
		_ = s.appointmentService.Delete(appt.AppointmentID)
		return nil, fmt.Errorf("failed to create order: %v", err)
	}

	return &BookingResponse{
		Appointment: appt,
		Order:       ord,
	}, nil
}

package bookingservice

import (
	"errors"
	"fmt"
	"hospital-service/internal/enums"
	"hospital-service/internal/models/appointment"
	"hospital-service/internal/models/order"
	"hospital-service/internal/services/appointmentservice"
	"hospital-service/internal/services/orderservice"
	"hospital-service/internal/websocket"
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
	wsHub              *websocket.Hub // ✅ Thêm WebSocket Hub
}

func NewBookingService(apptSvc *appointmentservice.AppointmentService, ordSvc *orderservice.OrderService, wsHub *websocket.Hub) *BookingService {
	return &BookingService{
		appointmentService: apptSvc,
		orderService:       ordSvc,
		wsHub:              wsHub,
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

	// ✅ 3. Broadcast notification đến bác sĩ qua WebSocket
	if s.wsHub != nil {
		go s.notifyDoctorNewAppointment(appt, ord)
	}

	return &BookingResponse{
		Appointment: appt,
		Order:       ord,
	}, nil
}

// notifyDoctorNewAppointment gửi notification đến bác sĩ khi có lịch hẹn mới
func (s *BookingService) notifyDoctorNewAppointment(appt *appointment.Appointment, ord *order.Order) {
	payload := map[string]interface{}{
		"appointment": appt,
		"order":       ord,
		"message":     "Bạn có lịch hẹn mới",
	}

	s.wsHub.BroadcastToDoctor(appt.DoctorID, websocket.NewAppointment, payload)
}

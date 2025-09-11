package appointmentservice

import (
	"errors"
	"fmt"
	"math/rand"
	"time"

	"hospital-service/internal/enums"
	"hospital-service/internal/models/appointment"
	"hospital-service/internal/repositories/appointmentrepo"

	"github.com/google/uuid"
)

type AppointmentService struct {
	repo *appointmentrepo.AppointmentRepo
}

func NewAppointmentService(repo *appointmentrepo.AppointmentRepo) *AppointmentService {
	return &AppointmentService{repo: repo}
}

// ---------------- Create ----------------
func (s *AppointmentService) Create(
	patientID, doctorID, hospitalID, slotID string,
	notes string,
	specialty enums.Specialty,
) (*appointment.Appointment, error) {

	// Kiểm tra các field bắt buộc
	if patientID == "" || doctorID == "" || hospitalID == "" || slotID == "" {
		return nil, errors.New("missing required fields: patient_id, doctor_id, hospital_id, slot_id")
	}

	existingAppointments, err := s.repo.FindByDoctorID(doctorID)
	if err != nil {
		return nil, fmt.Errorf("failed to check doctor's appointments: %v", err)
	}

	for _, a := range existingAppointments {
		if a.SlotID == slotID && a.Status != enums.Canceled {
			return nil, fmt.Errorf("doctor already has an appointment in this time slot")
		}
	}

	a := &appointment.Appointment{
		AppointmentID: generateAppointmentID(),
		PatientID:     patientID,
		DoctorID:      doctorID,
		HospitalID:    hospitalID,
		SlotID:        slotID,
		Notes:         &notes,
		Specialty:     specialty,
		CreatedAt:     time.Now(),
		UpdatedAt:     time.Now(),
		Status:        enums.Pending,
	}

	a.AppointmentCode = fmt.Sprintf("APPT-%d-%04d", time.Now().UnixNano(), rand.Intn(10000))

	if err := s.repo.Create(a); err != nil {
		return nil, err
	}

	return a, nil
}


// ---------------- GetByID ----------------
// Lấy lịch khám theo appointment_id
func (s *AppointmentService) GetByID(id string) (*appointment.Appointment, error) {
	return s.repo.GetByID(id)
}

// ---------------- GetByPatientID ----------------
// Lấy tất cả lịch khám theo patient_id
func (s *AppointmentService) GetByPatientID(patientID string) ([]appointment.Appointment, error) {
	return s.repo.FindByPatientID(patientID)
}

// ---------------- GetByDoctorID ----------------
// Lấy tất cả lịch khám theo doctor_id
func (s *AppointmentService) GetByDoctorID(doctorID string) ([]appointment.Appointment, error) {
	return s.repo.FindByDoctorID(doctorID)
}

// ---------------- UpdateStatus ----------------
// Cập nhật trạng thái lịch khám
func (s *AppointmentService) UpdateStatus(id string, status enums.AppointmentStatus) error {
	a, err := s.repo.GetByID(id)
	if err != nil {
		return fmt.Errorf("appointment not found: %v", err)
	}

	a.Status = status
	a.UpdatedAt = time.Now()
	return s.repo.Update(a)
}

// ---------------- UpdateDetail ----------------
// Cập nhật chi tiết lịch khám (ví dụ thay đổi giờ, ghi chú)
func (s *AppointmentService) UpdateDetail(id string, updatedData *appointment.Appointment) error {
	a, err := s.repo.GetByID(id)
	if err != nil {
		return fmt.Errorf("appointment not found: %v", err)
	}

	// Chỉ cập nhật các trường cho phép
	if updatedData.SlotID != "" {
		a.SlotID = updatedData.SlotID
	}
	if updatedData.Notes != nil {
		a.Notes = updatedData.Notes
	}
	a.UpdatedAt = time.Now()

	return s.repo.Update(a)
}

// ---------------- ListAll ----------------
// Lấy danh sách tất cả lịch khám (dùng cho admin)
func (s *AppointmentService) ListAll() ([]appointment.Appointment, error) {
	return s.repo.ListAll()
}

// ---------------- Delete ----------------
// Xóa lịch khám
func (s *AppointmentService) Delete(id string) error {
	return s.repo.Delete(id)
}


// ---------------- Helper ----------------
func generateAppointmentID() string {
	return uuid.NewString()
}

package appointmentservice

import (
	"errors"
	"fmt"
	"log"
	"math/rand"
	"time"

	"hospital-service/internal/enums"
	"hospital-service/internal/models/appointment"
	"hospital-service/internal/repositories/appointmentrepo"

	"github.com/google/uuid"
	"gorm.io/gorm"
)

type AppointmentService struct {
	repo         *appointmentrepo.AppointmentRepo
	timeSlotRepo *appointmentrepo.TimeSlotRepo
}

func NewAppointmentService(repo *appointmentrepo.AppointmentRepo, timeSlotRepo *appointmentrepo.TimeSlotRepo) *AppointmentService {
	return &AppointmentService{repo: repo, timeSlotRepo: timeSlotRepo}
}

// Tạo khám mới
func (s *AppointmentService) Create(
	patientID, doctorID, hospitalID, bookUserID string,
	slotIDs []string, notes string, serviceName string,
) (*appointment.Appointment, error) {

	if patientID == "" || doctorID == "" || hospitalID == "" || bookUserID == "" || len(slotIDs) == 0 {
		return nil, errors.New("missing required fields")
	}

	slots, err := s.timeSlotRepo.FindByIDs(slotIDs)

	if err != nil {
		log.Println("❌ [DEBUG ERROR] có lỗi xảy ra:", err)
		return nil, err
	}
	if len(slots) != len(slotIDs) {
		return nil, errors.New("some slots not found")
	}

	var status = enums.Pending
	if serviceName == "Tư vấn trực tuyến với bác sĩ" {
		status = enums.PendingOnline
	}

	var a *appointment.Appointment

	// Transaction bắt đầu
	err = s.repo.DB().Transaction(func(tx *gorm.DB) error {
		a = &appointment.Appointment{
			AppointmentID:   generateAppointmentID(),
			PatientID:       patientID,
			DoctorID:        doctorID,
			HospitalID:      hospitalID,
			BookUserId:      bookUserID,
			Notes:           &notes,
			ServiceName:     serviceName,
			CreatedAt:       time.Now(),
			UpdatedAt:       time.Now(),
			RelatedRecordID: nil,
			Status:          status,
		}
		a.AppointmentCode = fmt.Sprintf("APPT-%d-%04d", time.Now().UnixNano(), rand.Intn(10000))

		if err := tx.Create(a).Error; err != nil {
			log.Printf("❌ failed Save slot1")
			return err
		}

		// Gán appointment cho các slot
		for _, slot := range slots {
			if slot.AppointmentID != nil {
				return fmt.Errorf("slot %s is already booked", slot.SlotID)
			}
			slot.AppointmentID = &a.AppointmentID
			if err := tx.Save(&slot).Error; err != nil {
				log.Printf("❌ failed Save slot")
				return err
			}
		}
		return nil
	})

	if err != nil {
		log.Printf("❌ failed Save slot3")
		return nil, err
	}

	return s.repo.GetByID(a.AppointmentID)
}

// Tạo lịch tái khám
func (s *AppointmentService) CreateFollowUp(
	patientID, doctorID, hospitalID, bookUserID string,
	slotIDs []string, notes string, serviceName string,
	relatedRecordID string,
) (*appointment.Appointment, error) {

	if patientID == "" || doctorID == "" || hospitalID == "" || bookUserID == "" || len(slotIDs) == 0 || relatedRecordID == "" {
		return nil, errors.New("missing required fields")
	}

	slots, err := s.timeSlotRepo.FindByIDs(slotIDs)

	if err != nil {
		log.Println("❌ [DEBUG ERROR] có lỗi xảy ra:", err)
		return nil, err
	}
	if len(slots) != len(slotIDs) {
		return nil, errors.New("some slots not found")
	}

	var status = enums.Pending
	if serviceName == "Tư vấn trực tuyến với bác sĩ" {
		status = enums.PendingOnline
	}

	var a *appointment.Appointment

	// Transaction bắt đầu
	err = s.repo.DB().Transaction(func(tx *gorm.DB) error {
		a = &appointment.Appointment{
			AppointmentID:   generateAppointmentID(),
			PatientID:       patientID,
			DoctorID:        doctorID,
			HospitalID:      hospitalID,
			BookUserId:      bookUserID,
			Notes:           &notes,
			ServiceName:     serviceName,
			CreatedAt:       time.Now(),
			UpdatedAt:       time.Now(),
			RelatedRecordID: &relatedRecordID,
			Status:          status,
		}
		a.AppointmentCode = fmt.Sprintf("APPT-%d-%04d", time.Now().UnixNano(), rand.Intn(10000))

		if err := tx.Create(a).Error; err != nil {
			log.Printf("❌ failed Save slot1")
			return err
		}

		// Gán appointment cho các slot
		for _, slot := range slots {
			if slot.AppointmentID != nil {
				return fmt.Errorf("slot %s is already booked", slot.SlotID)
			}
			slot.AppointmentID = &a.AppointmentID
			if err := tx.Save(&slot).Error; err != nil {
				log.Printf("❌ failed Save slot")
				return err
			}
		}
		return nil
	})

	if err != nil {
		log.Printf("❌ failed Save slot3")
		return nil, err
	}

	return s.repo.GetByID(a.AppointmentID)
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

// ---------------- GetOnlineAppointments ----------------
// Lấy danh sách lịch khám trực tuyến (PendingOnline) theo bookUserID hoặc doctorID
func (s *AppointmentService) GetOnlineAppointments(bookUserID, doctorID string) ([]appointment.Appointment, error) {
	if bookUserID == "" && doctorID == "" {
		return nil, errors.New("bookUserID or doctorID is required")
	}

	var result []appointment.Appointment
	query := s.repo.DB().Model(&appointment.Appointment{})

	// Lọc theo bookUserID hoặc doctorID
	if bookUserID != "" {
		query = query.Where("book_user_id = ?", bookUserID)
	}
	if doctorID != "" {
		query = query.Where("doctor_id = ?", doctorID)
	}

	// Chỉ lấy các appointment có trạng thái PendingOnline
	query = query.Where("status = ?", enums.PendingOnline)

	query = query.
		Preload("Patient").
		Preload("Doctor").
		Preload("TimeSlots")

	if err := query.Find(&result).Error; err != nil {
		return nil, fmt.Errorf("failed to fetch online appointments: %v", err)
	}

	return result, nil
}

// ---------------- Helper ----------------
func generateAppointmentID() string {
	return uuid.NewString()
}

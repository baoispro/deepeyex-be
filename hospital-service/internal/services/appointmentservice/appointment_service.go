package appointmentservice

import (
	"errors"
	"fmt"
	"math/rand"
	"time"

	"hospital-service/internal/enums"
	"hospital-service/internal/models/appointment"
	"hospital-service/internal/models/doctor"
	"hospital-service/internal/repositories/appointmentrepo"
	"hospital-service/internal/repositories/doctorrepo"

	"github.com/google/uuid"
	"gorm.io/gorm"
)

type AppointmentService struct {
	repo         *appointmentrepo.AppointmentRepo
	timeSlotRepo *appointmentrepo.TimeSlotRepo
	doctorRepo   *doctorrepo.DoctorRepo
}

func NewAppointmentService(repo *appointmentrepo.AppointmentRepo, timeSlotRepo *appointmentrepo.TimeSlotRepo, doctorRepo *doctorrepo.DoctorRepo) *AppointmentService {
	return &AppointmentService{repo: repo, timeSlotRepo: timeSlotRepo, doctorRepo: doctorRepo}
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
			return err
		}

		// Gán appointment cho các slot
		for _, slot := range slots {
			if slot.AppointmentID != nil {
				return fmt.Errorf("slot %s is already booked", slot.SlotID)
			}
			slot.AppointmentID = &a.AppointmentID
			if err := tx.Save(&slot).Error; err != nil {
				return err
			}
		}
		return nil
	})

	if err != nil {
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
			return err
		}

		// Gán appointment cho các slot
		for _, slot := range slots {
			if slot.AppointmentID != nil {
				return fmt.Errorf("slot %s is already booked", slot.SlotID)
			}
			slot.AppointmentID = &a.AppointmentID
			if err := tx.Save(&slot).Error; err != nil {
				return err
			}
		}
		return nil
	})

	if err != nil {
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

// ---------------- GetByPatientIDWithFilters ----------------
// Lấy lịch khám theo patient_id với filters và sorting
func (s *AppointmentService) GetByPatientIDWithFilters(patientID, status, date, sortBy string) ([]appointment.Appointment, error) {
	if patientID == "" {
		return nil, errors.New("patient_id is required")
	}
	return s.repo.FindByPatientIDWithFilters(patientID, status, date, sortBy)
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
func (s *AppointmentService) ListAll(patientName, status, doctorID string) ([]appointment.Appointment, error) {
	return s.repo.FindWithFilters(patientName, status, doctorID)
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

// Lấy danh sách appointment hôm nay với slot đã sắp xếp
func (s *AppointmentService) GetTodayAppointments(doctorID string) ([]appointment.Appointment, error) {
	return s.repo.FindTodayAppointmentsByDoctor(doctorID)
}

// ---------------- CancelAppointment ----------------
// Hủy lịch khám với ràng buộc thời gian (không cho hủy nếu còn < 12 tiếng)
func (s *AppointmentService) CancelAppointment(id string) error {
	// Lấy appointment với TimeSlots
	appt, err := s.repo.GetByID(id)
	if err != nil {
		return fmt.Errorf("appointment not found: %v", err)
	}

	// Kiểm tra nếu appointment đã bị hủy hoặc đã hoàn thành
	if appt.Status == enums.Canceled {
		return errors.New("appointment is already canceled")
	}
	if appt.Status == enums.Completed || appt.Status == enums.CompletedOnline {
		return errors.New("cannot cancel completed appointment")
	}

	// Kiểm tra TimeSlots
	if len(appt.TimeSlots) == 0 {
		return errors.New("appointment has no time slots")
	}

	// Tìm slot có StartTime sớm nhất
	earliestSlot := appt.TimeSlots[0]
	for _, slot := range appt.TimeSlots {
		if slot.StartTime.Before(earliestSlot.StartTime) {
			earliestSlot = slot
		}
	}

	// Kiểm tra thời gian: nếu còn < 12 tiếng thì không cho hủy
	now := time.Now()
	timeUntilAppointment := earliestSlot.StartTime.Sub(now)
	const minCancelDuration = 12 * time.Hour

	if timeUntilAppointment < minCancelDuration {
		return fmt.Errorf("cannot cancel appointment within 12 hours of appointment time (%.1f hours remaining)", timeUntilAppointment.Hours())
	}

	// Transaction: Update status và giải phóng slots
	err = s.repo.DB().Transaction(func(tx *gorm.DB) error {
		// Update appointment status
		appt.Status = enums.Canceled
		appt.UpdatedAt = time.Now()
		if err := tx.Save(appt).Error; err != nil {
			return fmt.Errorf("failed to update appointment status: %v", err)
		}

		// Giải phóng các time slots
		for _, slot := range appt.TimeSlots {
			slot.AppointmentID = nil
			if err := tx.Save(&slot).Error; err != nil {
				return fmt.Errorf("failed to release slot %s: %v", slot.SlotID, err)
			}
		}

		return nil
	})

	if err != nil {
		return fmt.Errorf("failed to cancel appointment: %v", err)
	}

	return nil
}

// ---------------- EmergencyCancelAppointment ----------------
// Hủy appointment gấp và tự động chuyển sang bác sĩ thay thế
func (s *AppointmentService) EmergencyCancelAppointment(appointmentID, reason string) error {
	// Lấy appointment với TimeSlots
	appt, err := s.repo.GetByID(appointmentID)
	if err != nil {
		return fmt.Errorf("appointment not found: %v", err)
	}

	// Kiểm tra nếu appointment đã bị hủy hoặc đã hoàn thành
	if appt.Status == enums.Canceled {
		return errors.New("appointment is already canceled")
	}
	if appt.Status == enums.Completed || appt.Status == enums.CompletedOnline {
		return errors.New("cannot cancel completed appointment")
	}

	// Kiểm tra TimeSlots
	if len(appt.TimeSlots) == 0 {
		return errors.New("appointment has no time slots")
	}

	// Tìm slot có StartTime sớm nhất để kiểm tra thời gian
	earliestSlot := appt.TimeSlots[0]
	for _, slot := range appt.TimeSlots {
		if slot.StartTime.Before(earliestSlot.StartTime) {
			earliestSlot = slot
		}
	}

	// Với emergency cancel, cho phép hủy mọi lúc (không giới hạn thời gian)
	// Vì có thể có ca phẫu thuật, cấp cứu gấp, bác sĩ bị ốm đột xuất, etc.

	// Transaction: Tìm bác sĩ thay thế và chuyển appointment
	err = s.repo.DB().Transaction(func(tx *gorm.DB) error {
		// Tìm bác sĩ cùng chuyên khoa và bệnh viện có slot trống
		altDoctor, err := s.findReplacementDoctor(appt.DoctorID, earliestSlot.StartTime, earliestSlot.EndTime)
		if err != nil {
			return fmt.Errorf("failed to find replacement doctor: %v", err)
		}
		if altDoctor == nil {
			return errors.New("no available replacement doctor found with same specialty and hospital")
		}

		// Tìm slot phù hợp của bác sĩ thay thế
		replacementSlot, err := s.findReplacementSlot(altDoctor.DoctorID, earliestSlot.StartTime, earliestSlot.EndTime)
		if err != nil {
			return fmt.Errorf("failed to find replacement slot: %v", err)
		}
		if replacementSlot == nil {
			return fmt.Errorf("no available slot found for replacement doctor %s (%s) at %s-%s",
				altDoctor.DoctorID, altDoctor.FullName,
				earliestSlot.StartTime.Format("15:04"), earliestSlot.EndTime.Format("15:04"))
		}

		// Cập nhật appointment sang bác sĩ mới
		appt.DoctorID = altDoctor.DoctorID
		appt.Status = enums.Confirmed // Giữ nguyên status hoặc có thể set thành PENDING
		appt.UpdatedAt = time.Now()
		if err := tx.Save(appt).Error; err != nil {
			return fmt.Errorf("failed to update appointment: %v", err)
		}

		// Cập nhật slot cũ (giải phóng)
		for _, slot := range appt.TimeSlots {
			slot.AppointmentID = nil
			if err := tx.Save(&slot).Error; err != nil {
				return fmt.Errorf("failed to release slot %s: %v", slot.SlotID, err)
			}
		}

		// Gán appointment cho slot mới
		replacementSlot.AppointmentID = &appt.AppointmentID
		if err := tx.Save(replacementSlot).Error; err != nil {
			return fmt.Errorf("failed to assign appointment to replacement slot: %v", err)
		}

		return nil
	})

	if err != nil {
		return fmt.Errorf("failed to emergency cancel appointment: %v", err)
	}

	return nil
}

// Helper: Tìm bác sĩ thay thế
func (s *AppointmentService) findReplacementDoctor(originalDoctorID string, startTime, endTime time.Time) (*doctor.Doctor, error) {
	// Tìm bác sĩ cùng chuyên khoa và bệnh viện
	replacementDoctor, err := s.doctorRepo.FindBestReplacementDoctor(originalDoctorID, startTime, endTime)
	if err != nil {
		return nil, err
	}

	return replacementDoctor, nil
}

// Helper: Tìm slot thay thế
func (s *AppointmentService) findReplacementSlot(doctorID string, startTime, endTime time.Time) (*appointment.TimeSlot, error) {
	// Tìm slot cùng giờ
	slots, err := s.timeSlotRepo.FindByDoctorIDAndDateRange(doctorID, startTime, endTime)
	if err != nil {
		return nil, err
	}

	// Tìm slot trống cùng giờ
	for _, slot := range slots {
		if slot.AppointmentID == nil || *slot.AppointmentID == "" {
			if slot.StartTime.Equal(startTime) && slot.EndTime.Equal(endTime) {
				return &slot, nil
			}
		}
	}

	// Nếu không có slot cùng giờ, tìm slot cùng ca
	for _, slot := range slots {
		if slot.AppointmentID == nil || *slot.AppointmentID == "" {
			if s.isSameShift(startTime, slot.StartTime) {
				return &slot, nil
			}
		}
	}

	return nil, nil
}

// Helper: Kiểm tra cùng ca
func (s *AppointmentService) isSameShift(time1, time2 time.Time) bool {
	hour1 := time1.Hour()
	hour2 := time2.Hour()

	// Ca sáng: 6-12h
	if hour1 >= 6 && hour1 < 12 && hour2 >= 6 && hour2 < 12 {
		return true
	}
	// Ca chiều: 12-18h
	if hour1 >= 12 && hour1 < 18 && hour2 >= 12 && hour2 < 18 {
		return true
	}
	// Ca tối: 18-24h
	if hour1 >= 18 && hour1 < 24 && hour2 >= 18 && hour2 < 24 {
		return true
	}
	// Ca đêm: 0-6h
	if ((hour1 >= 0 && hour1 < 6) || (hour1 >= 18 && hour1 < 24)) &&
		((hour2 >= 0 && hour2 < 6) || (hour2 >= 18 && hour2 < 24)) {
		return true
	}

	return false
}

// ---------------- Helper ----------------
func generateAppointmentID() string {
	return uuid.NewString()
}

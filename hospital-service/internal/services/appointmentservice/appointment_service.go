package appointmentservice

import (
	"encoding/json"
	"errors"
	"fmt"
	"math/rand"
	"time"

	"hospital-service/internal/config"
	"hospital-service/internal/enums"
	"hospital-service/internal/models/appointment"
	"hospital-service/internal/models/doctor"
	"hospital-service/internal/models/patient"
	"hospital-service/internal/repositories/appointmentrepo"
	"hospital-service/internal/repositories/doctorrepo"
	"hospital-service/internal/repositories/patientrepo"
	"hospital-service/internal/websocket"

	"github.com/google/uuid"
	"gorm.io/gorm"
)

type AppointmentService struct {
	repo          *appointmentrepo.AppointmentRepo
	timeSlotRepo  *appointmentrepo.TimeSlotRepo
	doctorRepo    *doctorrepo.DoctorRepo
	patientRepo   *patientrepo.PatientRepo
	wsHub         *websocket.Hub
	pendingRepo   *appointmentrepo.PendingFollowUpRepo
	emailService  EmailServiceInterface
	cfg           config.Config
}

// EmailServiceInterface interface để tránh circular dependency
type EmailServiceInterface interface {
	SendFollowUpConfirmationEmail(patientEmail, patientName, doctorName, doctorFullName, hospitalName, confirmationLink, appointmentDate, appointmentTime string) error
}

func NewAppointmentService(cfg config.Config, repo *appointmentrepo.AppointmentRepo, timeSlotRepo *appointmentrepo.TimeSlotRepo, doctorRepo *doctorrepo.DoctorRepo, wsHub *websocket.Hub) *AppointmentService {
	return &AppointmentService{
		cfg:          cfg,
		repo:         repo,
		timeSlotRepo: timeSlotRepo,
		doctorRepo:   doctorRepo,
		wsHub:        wsHub,
	}
}

// SetPatientRepo set patient repo
func (s *AppointmentService) SetPatientRepo(patientRepo *patientrepo.PatientRepo) {
	s.patientRepo = patientRepo
}

// SetPendingRepo set pending follow-up repo
func (s *AppointmentService) SetPendingRepo(pendingRepo *appointmentrepo.PendingFollowUpRepo) {
	s.pendingRepo = pendingRepo
}

// SetEmailService set email service
func (s *AppointmentService) SetEmailService(emailService EmailServiceInterface) {
	s.emailService = emailService
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

	// Gửi notification cho bác sĩ biết lịch đã bị hủy
	if s.wsHub != nil {
		go s.notifyDoctorAppointmentCancelled(appt)
	}

	return nil
}

// notifyDoctorAppointmentCancelled gửi notification cho bác sĩ khi bệnh nhân hủy lịch
func (s *AppointmentService) notifyDoctorAppointmentCancelled(appt *appointment.Appointment) {
	payload := map[string]interface{}{
		"appointment_id":    appt.AppointmentID,
		"message":           "Bệnh nhân đã hủy lịch hẹn",
		"patient_id":        appt.PatientID,
		"status":            appt.Status,
		"notification_type": "APPOINTMENT_CANCELLED_BY_PATIENT",
	}

	// Broadcast đến doctor
	s.wsHub.BroadcastToDoctor(appt.DoctorID, websocket.CancelAppointment, payload)
	fmt.Printf("[Appointment Service] WebSocket notification sent to doctor %s about cancelled appointment %s\n", appt.DoctorID, appt.AppointmentID)
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

	// Với emergency cancel, chỉ cho phép hủy trong vòng 1 tiếng (sát giờ)
	// Nếu còn < 1 tiếng thì cho phép lễ tân/CSKH xử lý và tự động chuyển bác sĩ
	now := time.Now()
	timeUntilAppointment := earliestSlot.StartTime.Sub(now)
	const maxEmergencyDuration = 1 * time.Hour

	if timeUntilAppointment > maxEmergencyDuration {
		return fmt.Errorf("emergency cancel only allowed within 1 hour of appointment time (%.1f hours remaining)", timeUntilAppointment.Hours())
	}
	
	// Nếu quá sớm (còn > 1 tiếng) thì không cho emergency cancel, dùng cancel thường
	if timeUntilAppointment < 0 {
		return errors.New("appointment time has passed")
	}

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

// ---------------- CreatePendingFollowUp ----------------
// Tạo lịch tái khám pending và gửi email xác nhận cho bệnh nhân
func (s *AppointmentService) CreatePendingFollowUp(
	patientID, doctorID, hospitalID string,
	slotIDs []string, notes string, serviceName string, relatedRecordID *string,
) (*appointment.PendingFollowUpAppointment, error) {

	if patientID == "" || doctorID == "" || hospitalID == "" || len(slotIDs) == 0 {
		return nil, errors.New("missing required fields")
	}

	// Validate slots exist
	slots, err := s.timeSlotRepo.FindByIDs(slotIDs)
	if err != nil {
		return nil, fmt.Errorf("failed to find slots: %v", err)
	}
	if len(slots) != len(slotIDs) {
		return nil, errors.New("some slots not found")
	}

	// Check slots are available
	for _, slot := range slots {
		if slot.AppointmentID != nil {
			return nil, fmt.Errorf("slot %s is already booked", slot.SlotID)
		}
	}

	// Get patient info
	var patient *patient.Patient
	if s.patientRepo != nil {
		patient, err = s.patientRepo.FindByID(patientID)
		if err != nil {
			return nil, fmt.Errorf("patient not found: %v", err)
		}
	}

	// Get doctor info
	doctor, err := s.doctorRepo.FindByID(doctorID)
	if err != nil {
		return nil, fmt.Errorf("doctor not found: %v", err)
	}

	// Generate confirmation token
	token := generateConfirmationToken()

	// Marshal slot IDs to JSON
	slotIDsJSON, err := json.Marshal(slotIDs)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal slot IDs: %v", err)
	}

	// Create pending appointment
	pendingAppt := &appointment.PendingFollowUpAppointment{
		PendingID:        generateAppointmentID(),
		PatientID:        patientID,
		HospitalID:       hospitalID,
		DoctorID:         doctorID,
		SlotIDs:          string(slotIDsJSON),
		ServiceName:      serviceName,
		ConfirmationToken: token,
		Status:           "PENDING",
		Notes:            notes,
		ExpiresAt:        time.Now().Add(7 * 24 * time.Hour), // 7 days
		CreatedAt:        time.Now(),
		UpdatedAt:        time.Now(),
	}

	// Set RelatedRecordID if provided (optional)
	if relatedRecordID != nil && *relatedRecordID != "" {
		pendingAppt.RelatedRecordID = relatedRecordID
	}

	if err := s.pendingRepo.Create(pendingAppt); err != nil {
		return nil, fmt.Errorf("failed to create pending appointment: %v", err)
	}

	// Get appointment date and time from first slot
	appointmentDate := slots[0].StartTime.Format("02/01/2006")
	appointmentTime := fmt.Sprintf("%s - %s", 
		slots[0].StartTime.Format("15:04"), 
		slots[0].EndTime.Format("15:04"))

	// Send confirmation email to patient
	if s.emailService != nil && patient != nil {
		confirmationLink := fmt.Sprintf("%s/confirm-appointment?token=%s", s.cfg.FrontendURL, token)
		err = s.emailService.SendFollowUpConfirmationEmail(
			patient.Email,
			patient.FullName,
			doctor.FullName,
			doctor.FullName,
			hospitalID, // TODO: Get hospital name
			confirmationLink,
			appointmentDate,
			appointmentTime,
		)
		if err != nil {
			// Log error but don't fail the operation
			fmt.Printf("Failed to send confirmation email: %v\n", err)
		}
	}

	// Set additional info for response
	if patient != nil {
		pendingAppt.PatientName = patient.FullName
	}
	pendingAppt.DoctorName = doctor.FullName

	return pendingAppt, nil
}

// ---------------- ConfirmPendingFollowUp ----------------
// Xác nhận pending follow-up và tạo appointment thực sự
func (s *AppointmentService) ConfirmPendingFollowUp(token string) (*appointment.Appointment, error) {
	// Get pending appointment
	pendingAppt, err := s.pendingRepo.GetByToken(token)
	if err != nil {
		return nil, fmt.Errorf("invalid confirmation token")
	}

	// Check if already confirmed
	if pendingAppt.Status == "CONFIRMED" {
		return nil, errors.New("appointment already confirmed")
	}

	// Check if expired
	if time.Now().After(pendingAppt.ExpiresAt) {
		return nil, errors.New("confirmation token has expired")
	}

	// Parse slot IDs
	var slotIDs []string
	if err := json.Unmarshal([]byte(pendingAppt.SlotIDs), &slotIDs); err != nil {
		return nil, fmt.Errorf("invalid slot IDs format")
	}

	// Check slots are still available
	slots, err := s.timeSlotRepo.FindByIDs(slotIDs)
	if err != nil {
		return nil, fmt.Errorf("failed to find slots: %v", err)
	}

	for _, slot := range slots {
		if slot.AppointmentID != nil {
			return nil, fmt.Errorf("slot %s is no longer available", slot.SlotID)
		}
	}

	var createdAppointment *appointment.Appointment

	// Transaction: Create appointment and update pending status
	err = s.repo.DB().Transaction(func(tx *gorm.DB) error {
		// Create the actual appointment
		a := &appointment.Appointment{
			AppointmentID:   generateAppointmentID(),
			PatientID:      pendingAppt.PatientID,
			DoctorID:       pendingAppt.DoctorID,
			HospitalID:     pendingAppt.HospitalID,
			BookUserId:     "", // Use patient as book user
			Notes:          &pendingAppt.Notes,
			ServiceName:    pendingAppt.ServiceName,
			Status:         enums.Pending,
			CreatedAt:      time.Now(),
			UpdatedAt:      time.Now(),
		}
		a.AppointmentCode = fmt.Sprintf("APPT-%d-%04d", time.Now().UnixNano(), rand.Intn(10000))

		// Set RelatedRecordID if exists
		if pendingAppt.RelatedRecordID != nil && *pendingAppt.RelatedRecordID != "" {
			a.RelatedRecordID = pendingAppt.RelatedRecordID
		}

		if err := tx.Create(a).Error; err != nil {
			return err
		}

		// Assign slots to appointment
		for _, slot := range slots {
			slot.AppointmentID = &a.AppointmentID
			if err := tx.Save(&slot).Error; err != nil {
				return fmt.Errorf("failed to assign slot: %v", err)
			}
		}

		// Update pending appointment status
		now := time.Now()
		pendingAppt.Status = "CONFIRMED"
		pendingAppt.ConfirmedAt = &now
		if err := tx.Save(pendingAppt).Error; err != nil {
			return fmt.Errorf("failed to update pending appointment: %v", err)
		}

		createdAppointment = a
		return nil
	})

	if err != nil {
		return nil, fmt.Errorf("failed to confirm appointment: %v", err)
	}

	// Get full appointment with relations
	return s.repo.GetByID(createdAppointment.AppointmentID)
}

// generateConfirmationToken tạo token xác nhận ngẫu nhiên
func generateConfirmationToken() string {
	// Generate a random token using UUID and timestamp
	return fmt.Sprintf("%s-%d", uuid.NewString(), time.Now().Unix())
}

// ---------------- Helper ----------------
func generateAppointmentID() string {
	return uuid.NewString()
}

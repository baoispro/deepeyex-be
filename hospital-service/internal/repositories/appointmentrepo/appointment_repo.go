package appointmentrepo

import (
	"hospital-service/internal/models/appointment"
	"strings"
	"time"

	"gorm.io/gorm"
)

type AppointmentRepo struct {
	db *gorm.DB
}

func (r *AppointmentRepo) DB() *gorm.DB {
	return r.db
}

func NewAppointmentRepo(db *gorm.DB) *AppointmentRepo {
	return &AppointmentRepo{db: db}
}

// ---------------- Create ----------------
// Tạo mới một Appointment
func (r *AppointmentRepo) Create(a *appointment.Appointment) error {
	return r.db.Create(a).Error
}

// ---------------- FindByID ----------------
func (r *AppointmentRepo) GetByID(id string) (*appointment.Appointment, error) {
	var a appointment.Appointment
	err := r.db.
		Preload("Patient").
		Preload("TimeSlots").
		Preload("Doctor").
		First(&a, "appointment_id = ?", id).Error
	if err != nil {
		return nil, err
	}
	return &a, nil
}

// ---------------- FindByPatientID ----------------
// Lấy tất cả các Appointment theo patient_id
func (r *AppointmentRepo) FindByPatientID(patientID string) ([]appointment.Appointment, error) {
	var appointments []appointment.Appointment
	if err := r.db.Where("patient_id = ?", patientID).Preload("TimeSlots").
		Preload("TimeSlots.Doctor").
		Preload("Patient").Find(&appointments).Error; err != nil {
		return nil, err
	}
	return appointments, nil
}

// ---------------- FindByDoctorID ----------------
// Lấy tất cả các Appointment theo doctor_id
func (r *AppointmentRepo) FindByDoctorID(doctorID string) ([]appointment.Appointment, error) {
	var appointments []appointment.Appointment
	if err := r.db.Where("doctor_id = ?", doctorID).Preload("TimeSlots").
		Preload("TimeSlots.Doctor").
		Preload("Patient").Find(&appointments).Error; err != nil {
		return nil, err
	}
	return appointments, nil
}

// ---------------- Update ----------------
// Cập nhật thông tin Appointment
func (r *AppointmentRepo) Update(a *appointment.Appointment) error {
	return r.db.Save(a).Error
}

// ---------------- Delete ----------------
// Xóa Appointment theo appointment_id
func (r *AppointmentRepo) Delete(id string) error {
	return r.db.Delete(&appointment.Appointment{}, "appointment_id = ?", id).Error
}

// ---------------- List ----------------
func (r *AppointmentRepo) ListAll() ([]appointment.Appointment, error) {
	var appointments []appointment.Appointment
	err := r.db.
		Preload("Patient").
		Preload("TimeSlots").
		Preload("TimeSlots.Doctor").
		Find(&appointments).Error
	if err != nil {
		return nil, err
	}
	return appointments, nil
}

// FindWithFilters tìm appointments với filter động
func (r *AppointmentRepo) FindWithFilters(patientName, status, doctorID string) ([]appointment.Appointment, error) {
	var appointments []appointment.Appointment
	query := r.db.
		Preload("Patient").
		Preload("TimeSlots").
		Preload("TimeSlots.Doctor")

	// Filter theo patient name (partial match, case-insensitive)
	// Cần JOIN với bảng patients
	if patientName != "" {
		query = query.Joins("JOIN patients ON patients.patient_id = appointments.patient_id").
			Where("LOWER(patients.full_name) LIKE ?", "%"+strings.ToLower(patientName)+"%")
	}

	// Filter theo status (exact match)
	if status != "" {
		query = query.Where("appointments.status = ?", strings.ToUpper(status))
	}

	// Filter theo doctor_id (exact match)
	if doctorID != "" {
		query = query.Where("appointments.doctor_id = ?", doctorID)
	}

	if err := query.Find(&appointments).Error; err != nil {
		return nil, err
	}
	return appointments, nil
}

// Kiểm tra xem bác sĩ có appointment trong khoảng thời gian đó không
func (r *AppointmentRepo) ExistsByDoctorAndDateRange(doctorID string, start, end time.Time) (bool, error) {
	var count int64
	err := r.db.Model(&appointment.Appointment{}).
		Where("doctor_id = ? AND EXISTS (SELECT 1 FROM time_slots WHERE time_slots.appointment_id = appointments.appointment_id AND start_time < ? AND end_time > ?)",
			doctorID, end, start).
		Count(&count).Error
	if err != nil {
		return false, err
	}
	return count > 0, nil
}

// Cập nhật appointment sang bác sĩ khác
func (r *AppointmentRepo) ReassignDoctor(doctorID, newDoctorID string, start, end time.Time) error {
	return r.db.Transaction(func(tx *gorm.DB) error {
		var affected []appointment.Appointment

		// Lấy các appointment cần chuyển
		if err := tx.Model(&appointment.Appointment{}).
			Joins("JOIN time_slots ON time_slots.appointment_id = appointments.appointment_id").
			Where("appointments.doctor_id = ? AND time_slots.start_time < ? AND time_slots.end_time > ?", doctorID, end, start).
			Find(&affected).Error; err != nil {
			return err
		}

		if len(affected) == 0 {
			return nil
		}

		// Cập nhật doctor_id
		return tx.Model(&appointment.Appointment{}).
			Where("doctor_id = ? AND appointment_id IN (?)", doctorID, tx.Model(&appointment.Appointment{}).Select("appointment_id").
				Joins("JOIN time_slots ON time_slots.appointment_id = appointments.appointment_id").
				Where("time_slots.start_time < ? AND time_slots.end_time > ?", end, start)).
			Update("doctor_id", newDoctorID).Error
	})
}

func (r *AppointmentRepo) FindTodayAppointmentsByDoctor(doctorID string) ([]appointment.Appointment, error) {
	var appointments []appointment.Appointment

	err := r.db.Preload("TimeSlots", func(db *gorm.DB) *gorm.DB {
		return db.Order("start_time ASC")
	}).Joins("JOIN time_slots ON time_slots.appointment_id = appointments.appointment_id").
		Where("DATE(time_slots.start_time) = CURRENT_DATE AND appointments.doctor_id = ?", doctorID).
		Order("appointments.doctor_id ASC").
		Find(&appointments).Error

	return appointments, err
}

package appointmentrepo

import (
	"hospital-service/internal/models/appointment"

	"gorm.io/gorm"
)

type AppointmentRepo struct {
	db *gorm.DB
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
// Tìm Appointment theo appointment_id
func (r *AppointmentRepo) GetByID(id string) (*appointment.Appointment, error) {
	var a appointment.Appointment
	err := r.db.
		Preload("Patient").
		Preload("TimeSlot").
		Preload("TimeSlot.Doctor").
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
	if err := r.db.Where("patient_id = ?", patientID).Preload("TimeSlot").
		Preload("TimeSlot.Doctor").
		Preload("Patient").Find(&appointments).Error; err != nil {
		return nil, err
	}
	return appointments, nil
}

// ---------------- FindByDoctorID ----------------
// Lấy tất cả các Appointment theo doctor_id
func (r *AppointmentRepo) FindByDoctorID(doctorID string) ([]appointment.Appointment, error) {
	var appointments []appointment.Appointment
	if err := r.db.Where("doctor_id = ?", doctorID).Preload("TimeSlot").
		Preload("TimeSlot.Doctor").
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
		Preload("TimeSlot").
		Preload("TimeSlot.Doctor").
		Find(&appointments).Error
	if err != nil {
		return nil, err
	}
	return appointments, nil
}

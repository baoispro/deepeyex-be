package appointmentrepo

import (
	"hospital-service/internal/models/appointment"
	"time"

	"gorm.io/gorm"
)

type TimeSlotRepo struct {
	db *gorm.DB
}

func NewTimeSlotRepo(db *gorm.DB) *TimeSlotRepo {
	return &TimeSlotRepo{db: db}
}

// Create tạo mới một TimeSlot
func (r *TimeSlotRepo) Create(ts *appointment.TimeSlot) error {
	return r.db.Create(ts).Error
}

// CreateBatch tạo nhiều TimeSlot cùng lúc
func (r *TimeSlotRepo) CreateBatch(timeSlots []appointment.TimeSlot) error {
	return r.db.CreateInBatches(timeSlots, 100).Error
}

// FindByDoctorIDAndDateRange tìm TimeSlot theo doctor_id và khoảng thời gian
func (r *TimeSlotRepo) FindByDoctorIDAndDateRange(doctorID string, startDate, endDate time.Time) ([]appointment.TimeSlot, error) {
	var timeSlots []appointment.TimeSlot
	query := r.db.Where("start_time >= ? AND start_time < ?", startDate, endDate)
	
	// Chỉ thêm điều kiện doctor_id nếu không rỗng
	if doctorID != "" {
		query = query.Where("doctor_id = ?", doctorID)
	}
	
	err := query.Find(&timeSlots).Error
	return timeSlots, err
}

// FindByDoctorIDAndDate tìm TimeSlot theo doctor_id và ngày cụ thể
func (r *TimeSlotRepo) FindByDoctorIDAndDate(doctorID string, date time.Time) ([]appointment.TimeSlot, error) {
	startOfDay := time.Date(date.Year(), date.Month(), date.Day(), 0, 0, 0, 0, date.Location())
	endOfDay := startOfDay.Add(24 * time.Hour)
	return r.FindByDoctorIDAndDateRange(doctorID, startOfDay, endOfDay)
}

// DeleteByDoctorIDAndDateRange xóa TimeSlot theo doctor_id và khoảng thời gian
func (r *TimeSlotRepo) DeleteByDoctorIDAndDateRange(doctorID string, startDate, endDate time.Time) error {
	return r.db.Where("doctor_id = ? AND start_time >= ? AND start_time < ?", doctorID, startDate, endDate).
		Delete(&appointment.TimeSlot{}).Error
}

// DeleteByDoctorID xóa toàn bộ timeslot của 1 bác sĩ
func (r *TimeSlotRepo) DeleteByDoctorID(doctorID string) error {
    return r.db.Where("doctor_id = ?", doctorID).Delete(&appointment.TimeSlot{}).Error
}

// FindAvailableSlots tìm các slot còn trống
func (r *TimeSlotRepo) FindAvailableSlots(doctorID string, startDate, endDate time.Time) ([]appointment.TimeSlot, error) {
	var timeSlots []appointment.TimeSlot
	err := r.db.Where("doctor_id = ? AND start_time >= ? AND start_time < ? AND appointment_id IS NULL", 
		doctorID, startDate, endDate).Find(&timeSlots).Error
	return timeSlots, err
}

// FindByID tìm TimeSlot theo ID
func (r *TimeSlotRepo) FindByID(id string) (*appointment.TimeSlot, error) {
	var timeSlot appointment.TimeSlot
	err := r.db.First(&timeSlot, "slot_id = ?", id).Error
	if err != nil {
		return nil, err
	}
	return &timeSlot, nil
}

// FindByDoctorID tìm TimeSlot theo doctor ID
func (r *TimeSlotRepo) FindByDoctorID(doctorID string) ([]appointment.TimeSlot, error) {
	var timeSlots []appointment.TimeSlot
	err := r.db.Where("doctor_id = ?", doctorID).Find(&timeSlots).Error
	return timeSlots, err
}

// Update cập nhật TimeSlot
func (r *TimeSlotRepo) Update(timeSlot *appointment.TimeSlot) error {
	return r.db.Save(timeSlot).Error
}

// Delete xóa TimeSlot theo ID
func (r *TimeSlotRepo) Delete(id string) error {
	return r.db.Delete(&appointment.TimeSlot{}, "slot_id = ?", id).Error
}

// ListAll lấy tất cả TimeSlot
func (r *TimeSlotRepo) ListAll() ([]appointment.TimeSlot, error) {
	var timeSlots []appointment.TimeSlot
	err := r.db.Preload("Doctor").Find(&timeSlots).Error
	return timeSlots, err
}

// CountOverlapping đếm số slot trùng thời gian
func (r *TimeSlotRepo) CountOverlapping(doctorID string, startTime, endTime time.Time) (int64, error) {
	var count int64
	err := r.db.Model(&appointment.TimeSlot{}).
		Where("doctor_id = ? AND NOT (end_time <= ? OR start_time >= ?)", doctorID, startTime, endTime).
		Count(&count).Error
	return count, err
}

// FindByDoctorAndDate tìm TimeSlot theo doctor và ngày
func (r *TimeSlotRepo) FindByDoctorAndDate(doctorID string, date time.Time) ([]appointment.TimeSlot, error) {
	startOfDay := time.Date(date.Year(), date.Month(), date.Day(), 0, 0, 0, 0, date.Location())
	endOfDay := startOfDay.Add(24 * time.Hour)

	var timeSlots []appointment.TimeSlot
	err := r.db.Where("doctor_id = ? AND start_time >= ? AND start_time < ?", doctorID, startOfDay, endOfDay).
		Order("start_time asc").
		Find(&timeSlots).Error
	return timeSlots, err
}

// FindByDoctorAndMonth tìm TimeSlot theo doctor và tháng
func (r *TimeSlotRepo) FindByDoctorAndMonth(doctorID string, date time.Time) ([]appointment.TimeSlot, error) {
	startOfMonth := time.Date(date.Year(), date.Month(), 1, 0, 0, 0, 0, date.Location())
	nextMonth := startOfMonth.AddDate(0, 1, 0)

	var timeSlots []appointment.TimeSlot
	err := r.db.Where("doctor_id = ? AND start_time >= ? AND start_time < ?", doctorID, startOfMonth, nextMonth).
		Order("start_time asc").
		Find(&timeSlots).Error
	return timeSlots, err
}

// FindByIDs tìm TimeSlot theo danh sách ID
func (r *TimeSlotRepo) FindByIDs(ids []string) ([]appointment.TimeSlot, error) {
	var timeSlots []appointment.TimeSlot
	err := r.db.Where("slot_id IN ?", ids).Find(&timeSlots).Error
	return timeSlots, err
}

//// FindByDoctorAndMonth tìm TimeSlot theo doctor và từ ngày startDate đến endDate
func (r *TimeSlotRepo) FindByDoctorAndDateRange(doctorID string, startDate, endDate time.Time) ([]appointment.TimeSlot, error) {
	var timeSlots []appointment.TimeSlot
	err := r.db.Where("doctor_id = ? AND start_time >= ? AND start_time < ?", doctorID, startDate, endDate).
		Order("start_time asc").
		Find(&timeSlots).Error
	return timeSlots, err
}


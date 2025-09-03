package hospitalrepo

import (
	"hospital-service/internal/models/hospital"

	"gorm.io/gorm"
)

// HospitalRepo quản lý truy cập dữ liệu cho Hospital
type HospitalRepo struct {
	db *gorm.DB
}

// NewHospitalRepo khởi tạo repository
func NewHospitalRepo(db *gorm.DB) *HospitalRepo {
	return &HospitalRepo{db: db}
}

// Create hospital
func (r *HospitalRepo) Create(h *hospital.Hospital) error {
	return r.db.Create(h).Error
}

// FindByID tìm hospital theo ID
func (r *HospitalRepo) FindByID(id string) (*hospital.Hospital, error) {
	var h hospital.Hospital
	err := r.db.First(&h, "hospital_id = ?", id).Error
	return &h, err
}

// Update hospital
func (r *HospitalRepo) Update(h *hospital.Hospital) error {
	return r.db.Save(h).Error
}

// Delete hospital
func (r *HospitalRepo) Delete(id string) error {
	return r.db.Delete(&hospital.Hospital{}, "hospital_id = ?", id).Error
}

// List hospitals
func (r *HospitalRepo) List() ([]hospital.Hospital, error) {
	var hospitals []hospital.Hospital
	err := r.db.Find(&hospitals).Error
	return hospitals, err
}

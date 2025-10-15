package medicalrecordrepo

import (
	"gorm.io/gorm"
	"hospital-service/internal/models/medicalrecord"
)

type AIDiagnosisRepo struct {
	db *gorm.DB
}

// NewAIDiagnosisRepo khởi tạo repository
func NewAIDiagnosisRepo(db *gorm.DB) *AIDiagnosisRepo {
	return &AIDiagnosisRepo{db: db}
}

// Create tạo mới một bản ghi chẩn đoán AI
func (r *AIDiagnosisRepo) Create(d *medicalrecord.AIDiagnosis) error {
	return r.db.Create(d).Error
}

// FindAll lấy toàn bộ bản ghi chẩn đoán AI
func (r *AIDiagnosisRepo) FindAllPending() ([]medicalrecord.AIDiagnosis, error) {
	var diagnoses []medicalrecord.AIDiagnosis
	if err := r.db.Where("status = ?", "PENDING").Order("created_at DESC").Find(&diagnoses).Error; err != nil {
		return nil, err
	}
	return diagnoses, nil
}
// FindByPatientID tìm danh sách chẩn đoán AI theo PatientID
func (r *AIDiagnosisRepo) FindByPatientID(patientID string) ([]medicalrecord.AIDiagnosis, error) {
	var diagnoses []medicalrecord.AIDiagnosis
	if err := r.db.Where("patient_id = ?", patientID).Order("created_at DESC").Find(&diagnoses).Error; err != nil {
		return nil, err
	}
	return diagnoses, nil
}

// FindByID tìm chẩn đoán theo ID
func (r *AIDiagnosisRepo) FindByID(id string) (*medicalrecord.AIDiagnosis, error) {
	var diagnosis medicalrecord.AIDiagnosis
	if err := r.db.First(&diagnosis, "id = ?", id).Error; err != nil {
		return nil, err
	}
	return &diagnosis, nil
}

// Update cập nhật thông tin chẩn đoán (ví dụ khi bác sĩ xác nhận)
func (r *AIDiagnosisRepo) Update(d *medicalrecord.AIDiagnosis) error {
	return r.db.Save(d).Error
}

// Delete xoá bản ghi chẩn đoán AI
func (r *AIDiagnosisRepo) Delete(id string) error {
	return r.db.Delete(&medicalrecord.AIDiagnosis{}, "id = ?", id).Error
}

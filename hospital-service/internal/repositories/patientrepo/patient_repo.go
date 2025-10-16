package patientrepo

import (
	"hospital-service/internal/models/patient"
	"strings"

	"gorm.io/gorm"
)

type PatientRepo struct {
	db *gorm.DB
}

func NewPatientRepo(db *gorm.DB) *PatientRepo {
	return &PatientRepo{db: db}
}

// Create thêm patient mới
func (r *PatientRepo) Create(p *patient.Patient) error {
	return r.db.Create(p).Error
}

// FindByUserID tìm patient theo UserID từ AuthService
func (r *PatientRepo) FindByUserID(userID string) (*patient.Patient, error) {
	var p patient.Patient
	if err := r.db.Where("user_id = ?", userID).First(&p).Error; err != nil {
		return nil, err
	}
	return &p, nil
}

// FindByID tìm patient theo PatientID
func (r *PatientRepo) FindByID(id string) (*patient.Patient, error) {
	var p patient.Patient
	if err := r.db.First(&p, "patient_id = ?", id).Error; err != nil {
		return nil, err
	}
	return &p, nil
}

// Update cập nhật thông tin patient
func (r *PatientRepo) Update(p *patient.Patient) error {
	return r.db.Save(p).Error
}

// Delete xóa patient
func (r *PatientRepo) Delete(id string) error {
	return r.db.Delete(&patient.Patient{}, "patient_id = ?", id).Error
}

// List tất cả patients
func (r *PatientRepo) List() ([]patient.Patient, error) {
	var patients []patient.Patient
	if err := r.db.Find(&patients).Error; err != nil {
		return nil, err
	}
	return patients, nil
}

// FindWithFilters tìm patients với filter động
func (r *PatientRepo) FindWithFilters(name, gender, birthDate string) ([]patient.Patient, error) {
	var patients []patient.Patient
	query := r.db

	// Filter theo name (partial match, case-insensitive)
	if name != "" {
		query = query.Where("LOWER(full_name) LIKE ?", "%"+strings.ToLower(name)+"%")
	}

	// Filter theo gender (exact match)
	if gender != "" {
		query = query.Where("gender = ?", gender)
	}

	// Filter theo birth date (format: YYYY-MM)
	if birthDate != "" {
		parts := strings.Split(birthDate, "-")
		if len(parts) == 2 {
			year := parts[0]
			month := parts[1]
			query = query.Where("EXTRACT(YEAR FROM dob) = ? AND EXTRACT(MONTH FROM dob) = ?", year, month)
		}
	}

	if err := query.Find(&patients).Error; err != nil {
		return nil, err
	}
	return patients, nil
}

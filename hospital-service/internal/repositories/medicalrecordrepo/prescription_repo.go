package medicalrecordrepo

import (
	"hospital-service/internal/models/medicalrecord"
	"time"

	"gorm.io/gorm"
)

type PrescriptionRepo struct {
	db *gorm.DB
}

func NewPrescriptionRepository(db *gorm.DB) *PrescriptionRepo {
	return &PrescriptionRepo{db: db}
}

// ---------------- Create ----------------
func (r *PrescriptionRepo) Create(prescription *medicalrecord.Prescription) error {
	return r.db.Create(prescription).Error
}

// --------------- Get Prescription By ID ----------------
func (r *PrescriptionRepo) GetPrescriptionByID(id string) (*medicalrecord.Prescription, error) {
	var p medicalrecord.Prescription
	err := r.db.Preload("Items").First(&p, "prescription_id = ?", id).Error
	return &p, err
}

// --------------- Get Prescriptions By Medical Record ID ----------------
func (r *PrescriptionRepo) GetPrescriptionsByMedicalRecordID(medicalRecordID string) ([]*medicalrecord.Prescription, error) {
	var prescriptions []*medicalrecord.Prescription
	err := r.db.Preload("Items").Where("medical_record_id = ?", medicalRecordID).Find(&prescriptions).Error
	return prescriptions, err
}

// --------------- Get Prescriptions By Patient ID ----------------
func (r *PrescriptionRepo) GetPrescriptionsByPatientID(patientID string) ([]*medicalrecord.Prescription, error) {
	var prescriptions []*medicalrecord.Prescription
	err := r.db.Preload("Items").Where("patient_id = ?", patientID).
		Order("created_at DESC").
		Find(&prescriptions).Error
	return prescriptions, err
}

// --------------- Get Prescriptions By Patient ID With Filters ----------------
func (r *PrescriptionRepo) FindByPatientIDWithFilters(patientID, status, date, sortBy string) ([]*medicalrecord.Prescription, error) {
	var prescriptions []*medicalrecord.Prescription

	query := r.db.Where("patient_id = ?", patientID).
		Preload("Items")

	// Filter theo status
	if status != "" {
		query = query.Where("status = ?", status)
	}

	// Filter theo ngày tạo (created_at)
	if date != "" {
		// Parse date string (expected format: YYYY-MM-DD)
		parsedDate, err := time.Parse("2006-01-02", date)
		if err == nil {
			// Filter theo ngày tạo prescription
			query = query.Where("DATE(created_at) = ?", parsedDate.Format("2006-01-02"))
		}
	}

	// Sort theo created_at
	switch sortBy {
	case "newest":
		query = query.Order("created_at DESC")
	case "oldest":
		query = query.Order("created_at ASC")
	default:
		// Default: sort theo mới nhất
		query = query.Order("created_at DESC")
	}

	if err := query.Find(&prescriptions).Error; err != nil {
		return nil, err
	}

	return prescriptions, nil
}

// --------------- Update Prescription ----------------
func (r *PrescriptionRepo) Approve(id, doctorID string) error {
	return r.db.Model(&medicalrecord.Prescription{}).Where("prescription_id = ?", id).
		Updates(map[string]interface{}{
			"status":      "APPROVED",
			"approved_by": doctorID,
			"approved_at": time.Now(),
		}).Error
}

// --------------- Update Prescription By ID ----------------
func (r *PrescriptionRepo) UpdatePrecription(p *medicalrecord.Prescription) error {
	return r.db.Save(p).Error
}

// --------------- Detele Prescription  ----------------
func (r *PrescriptionRepo) Delete(id string) error {
	return r.db.Delete(&medicalrecord.Prescription{}, "prescription_id = ?", id).Error
}

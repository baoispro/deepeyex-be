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
	err := r.db.Preload("Items").Where("record_id = ?", medicalRecordID).Find(&prescriptions).Error
	return prescriptions, err
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

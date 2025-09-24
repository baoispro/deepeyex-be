package medicalrecordrepo

import (
	"hospital-service/internal/models/medicalrecord"

	"gorm.io/gorm"
)


type PrescriptionItemRepo struct {
	db *gorm.DB
}

func NewPrescriptionItemRepository(db *gorm.DB) *PrescriptionItemRepo {
	return &PrescriptionItemRepo{db: db}
}

// ---------------- Create ----------------
func (r *PrescriptionItemRepo) Create(p *medicalrecord.Prescription) error {
	return r.db.Create(p).Error
}

// ---------------- Update Prescription Item ----------------
func (r *PrescriptionItemRepo) UpdatePrescriptionItem(item *medicalrecord.PrescriptionItem) error {
	return r.db.Save(item).Error
}

// ---------------- Delete Prescription Item ----------------
func (r *PrescriptionItemRepo) Delete(id string) error {
	return r.db.Delete(&medicalrecord.PrescriptionItem{}, "prescription_item_id = ?", id).Error
}
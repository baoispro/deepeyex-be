package medicalrecordservice

import (
	"errors"
	"hospital-service/internal/models/medicalrecord"
	"hospital-service/internal/repositories/medicalrecordrepo"
)

type PrescriptionItemService struct {
	repo *medicalrecordrepo.PrescriptionItemRepo
}

// NewPrescriptionItemService khởi tạo service
func NewPrescriptionItemService(repo *medicalrecordrepo.PrescriptionItemRepo) *PrescriptionItemService {
	return &PrescriptionItemService{repo: repo}
}

// ---------------- Create ----------------
func (s *PrescriptionItemService) CreatePrescriptionItem(prescriptionID, drugName, dosage, frequency string, durationDays int) (*medicalrecord.PrescriptionItem, error) {
	// Validate các trường bắt buộc
	if prescriptionID == "" || drugName == "" || dosage == "" || frequency == "" {
		return nil, errors.New("missing required fields: prescription_id, drug_name, dosage, frequency")
	}

	item := &medicalrecord.PrescriptionItem{
		PrescriptionID: prescriptionID,
		DrugName:       drugName,
		Dosage:         dosage,
		Frequency:      frequency,
		DurationDays:   durationDays,
	}

	// Tạo mới trong DB
	err := s.repo.Create(&medicalrecord.Prescription{
		PrescriptionID: prescriptionID,
		Items:          []medicalrecord.PrescriptionItem{*item},
	})
	if err != nil {
		return nil, err
	}
	return item, nil
}

// ---------------- Update ----------------
func (s *PrescriptionItemService) UpdatePrescriptionItem(item *medicalrecord.PrescriptionItem) error {
	if item.ItemID == "" {
		return errors.New("missing prescription_item_id")
	}
	return s.repo.UpdatePrescriptionItem(item)
}

// ---------------- Delete ----------------
func (s *PrescriptionItemService) DeletePrescriptionItem(itemID string) error {
	if itemID == "" {
		return errors.New("missing prescription_item_id")
	}
	return s.repo.Delete(itemID)
}

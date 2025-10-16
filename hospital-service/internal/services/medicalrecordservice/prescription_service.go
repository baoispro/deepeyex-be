package medicalrecordservice

import (
	"errors"
	"hospital-service/internal/models/medicalrecord"
	"hospital-service/internal/repositories/medicalrecordrepo"
	"time"

	"github.com/google/uuid"
)

type PrescriptionService struct {
	repo                   *medicalrecordrepo.PrescriptionRepo
	prescriptionItemRepo   *medicalrecordrepo.PrescriptionItemRepo
	medicationReminderRepo *medicalrecordrepo.MedicationReminderRepository
}

func NewPrescriptionService(repo *medicalrecordrepo.PrescriptionRepo, itemRepo *medicalrecordrepo.PrescriptionItemRepo,
	reminderRepo *medicalrecordrepo.MedicationReminderRepository) *PrescriptionService {
	return &PrescriptionService{repo: repo, prescriptionItemRepo: itemRepo,
		medicationReminderRepo: reminderRepo}
}

// ---------------- Request struct ----------------
type PrescriptionRequest struct {
	PatientID       string                    `json:"patient_id"`
	MedicalRecordID *string                   `json:"medical_record_id,omitempty"`
	Source          string                    `json:"source"`
	Description     *string                   `json:"description,omitempty"`
	Items           []PrescriptionItemRequest `json:"items"`
}

type PrescriptionItemRequest struct {
	DrugName     string    `json:"drug_name"`
	Dosage       string    `json:"dosage"`
	Frequency    string    `json:"frequency"` // nhập "3", "2", ...
	DurationDays int       `json:"duration_days"`
	StartDate    time.Time `json:"start_date"`
	Notes        *string   `json:"notes,omitempty"`
	CustomTimes  []string  `json:"custom_times"`
}

type CreatePrescriptionRequest struct {
	AI_DiagnosisID  *string                   `json:"ai_diagnosis_id,omitempty"`
	MedicalRecordID *string                   `json:"medical_record_id,omitempty"`
	PatientID       string                    `json:"patient_id"`
	Source          string                    `json:"source"` // AI or DOCTOR
	Description     *string                   `json:"description,omitempty"`
	Items           []PrescriptionItemRequest `json:"items"`
}

// ---------------- CreatePrescription ----------------
func (s *PrescriptionService) CreatePrescription(req *CreatePrescriptionRequest) error {
	if len(req.Items) == 0 {
		return errors.New("at least one prescription item is required")
	}

	// 🩺 Tạo Prescription
	prescription := &medicalrecord.Prescription{
		PrescriptionID:  uuid.NewString(),
		AI_DiagnosisID:  req.AI_DiagnosisID,
		MedicalRecordID: req.MedicalRecordID,
		PatientID:       req.PatientID,
		Source:          req.Source,
		Description:     req.Description,
		Status:          "PENDING",
		CreatedAt:       time.Now(),
		UpdatedAt:       time.Now(),
	}

	if err := s.repo.Create(prescription); err != nil {
		return errors.New("failed to create prescription")
	}

	// 🧾 Từng thuốc trong toa
	for _, itemReq := range req.Items {
		itemID := uuid.NewString()
		endDate := itemReq.StartDate.AddDate(0, 0, itemReq.DurationDays)

		item := &medicalrecord.PrescriptionItem{
			ItemID:         itemID,
			PrescriptionID: prescription.PrescriptionID,
			DrugName:       itemReq.DrugName,
			Dosage:         itemReq.Dosage,
			Frequency:      itemReq.Frequency,
			DurationDays:   itemReq.DurationDays,
			Notes:          itemReq.Notes,
			StartDate:      itemReq.StartDate,
			EndDate:        endDate,
		}

		if err := s.prescriptionItemRepo.Create(item); err != nil {
			return errors.New("failed to create prescription item")
		}

		// 🕒 Tạo reminder cho từng thuốc
		err := s.createRemindersForItem(itemID, itemReq)
		if err != nil {
			return errors.New("failed to create reminders")
		}
	}

	return nil
}

// 🔹 Sub-function: tạo reminders theo auto/manual
func (s *PrescriptionService) createRemindersForItem(itemID string, itemReq PrescriptionItemRequest) error {
	for day := 0; day < itemReq.DurationDays; day++ {
		currentDate := itemReq.StartDate.AddDate(0, 0, day)

		// 🧩 Case 1: Manual times (FE gửi custom_times)
		if len(itemReq.CustomTimes) > 0 {
			for _, t := range itemReq.CustomTimes {
				parsedTime, err := time.Parse("15:04", t)
				if err != nil {
					continue // skip invalid time format
				}
				reminderTime := time.Date(
					currentDate.Year(), currentDate.Month(), currentDate.Day(),
					parsedTime.Hour(), parsedTime.Minute(), 0, 0, currentDate.Location(),
				)
				reminder := &medicalrecord.MedicationReminder{
					ID:                 uuid.NewString(),
					PrescriptionItemID: itemID,
					ReminderTime:       reminderTime,
					Status:             "PENDING",
				}
				if err := s.medicationReminderRepo.Create(reminder); err != nil {
					return err
				}
			}
			continue
		}
	}

	return nil
}

func (s *PrescriptionService) GetPrescriptionByID(id string) (*medicalrecord.Prescription, error) {
	return s.repo.GetPrescriptionByID(id)
}

func (s *PrescriptionService) GetPrescriptionsByMedicalRecordID(medicalRecordID string) ([]*medicalrecord.Prescription, error) {
	return s.repo.GetPrescriptionsByMedicalRecordID(medicalRecordID)
}

func (s *PrescriptionService) Approve(id, doctorID string) error {
	return s.repo.Approve(id, doctorID)
}

func (s *PrescriptionService) UpdatePrecription(p *medicalrecord.Prescription) error {
	return s.repo.UpdatePrecription(p)
}

func (s *PrescriptionService) Delete(id string) error {
	return s.repo.Delete(id)
}

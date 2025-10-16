package fullrecordservice

import (
	"errors"
	"hospital-service/internal/enums"
	"hospital-service/internal/models/medicalrecord"
	"hospital-service/internal/services/appointmentservice"
	"hospital-service/internal/services/medicalrecordservice"
	"time"
)

type FullRecordService struct {
	medicalRecordService *medicalrecordservice.MedicalRecordService
	attachmentService    *medicalrecordservice.AttachmentService
	prescriptionService  *medicalrecordservice.PrescriptionService
	appointmentService   *appointmentservice.AppointmentService
}

func NewFullRecordService(
	recordService *medicalrecordservice.MedicalRecordService,
	attachmentService *medicalrecordservice.AttachmentService,
	prescriptionService *medicalrecordservice.PrescriptionService,
	appointmentService *appointmentservice.AppointmentService,
) *FullRecordService {
	return &FullRecordService{
		medicalRecordService: recordService,
		attachmentService:    attachmentService,
		prescriptionService:  prescriptionService,
		appointmentService:   appointmentService,
	}
}

// Struct nhận dữ liệu đầu vào
type FullRecordCreateRequest struct {
	PatientID       string                                          `json:"patient_id"`
	DoctorID        string                                          `json:"doctor_id"`
	Diagnosis       string                                          `json:"diagnosis"`
	AppointmentID   string                                          `json:"appointment_id"`
	Notes           *string                                         `json:"notes,omitempty"`
	RelatedRecordID *string                                         `json:"related_record_id,omitempty"`
	Attachments     []AttachmentRequest                             `json:"attachments"`
	Prescription    *medicalrecordservice.CreatePrescriptionRequest `json:"prescription,omitempty"`
}

type AttachmentRequest struct {
	FileType string      `json:"file_type"`
	File     interface{} `json:"file"`
}

type CompleteRecordRequest struct {
	RecordID      string                                          `json:"record_id"`
	Diagnosis     string                                          `json:"diagnosis"`
	Notes         *string                                         `json:"notes,omitempty"`
	Attachments   []AttachmentRequest                             `json:"attachments"`
	Prescription  *medicalrecordservice.CreatePrescriptionRequest `json:"prescription,omitempty"`
	UpdateDoctor  *string                                         `json:"update_doctor,omitempty"` // tùy chọn: cập nhật doctor nếu khác
	UpdatePatient *string                                         `json:"update_patient,omitempty"`
}

func (s *FullRecordService) CreateFullRecord(req *FullRecordCreateRequest) (*medicalrecord.MedicalRecord, error) {
	if req.PatientID == "" || req.DoctorID == "" || req.Diagnosis == "" {
		return nil, errors.New("missing required fields: patient_id, doctor_id, diagnosis")
	}

	// 🩺 1️⃣ Tạo medical record
	record, err := s.medicalRecordService.CreateRecord(
		req.PatientID,
		req.DoctorID,
		req.Diagnosis,
		req.AppointmentID,
		req.Notes,
		req.RelatedRecordID,
	)
	if err != nil {
		return nil, err
	}

	// 📎 2️⃣ Tạo attachments (nếu có)
	for _, attReq := range req.Attachments {
		att := &medicalrecord.Attachment{
			RecordID: record.RecordID,
			FileType: attReq.FileType,
		}

		if _, err := s.attachmentService.AddAttachment(att, attReq.File); err != nil {
			return nil, err
		}
	}

	// 💊 3️⃣ Tạo prescription (nếu có)
	if req.Prescription != nil {
		req.Prescription.MedicalRecordID = &record.RecordID
		req.Prescription.PatientID = req.PatientID

		if err := s.prescriptionService.CreatePrescription(req.Prescription); err != nil {
			return nil, err
		}
	}

	// ✅ 4️⃣ Update appointment status thành COMPLETED (nếu có appointment_id)
	if req.AppointmentID != "" {
		if err := s.appointmentService.UpdateStatus(req.AppointmentID, enums.Completed); err != nil {
			// Log error nhưng không fail toàn bộ request
			// Vì medical record đã được tạo thành công
			// Có thể log hoặc return warning
		}
	}

	return record, nil
}

func (s *FullRecordService) CompleteRecord(req *CompleteRecordRequest) (*medicalrecord.MedicalRecord, error) {
	if req.RecordID == "" || req.Diagnosis == "" {
		return nil, errors.New("record_id and diagnosis are required")
	}

	// 🔍 Lấy record hiện tại
	record, err := s.medicalRecordService.GetRecord(req.RecordID)
	if err != nil {
		return nil, errors.New("medical record not found")
	}

	// 🩺 1️⃣ Cập nhật diagnosis, notes, doctor/patient (nếu có)
	record.Diagnosis = req.Diagnosis
	record.Notes = req.Notes
	record.UpdatedAt = time.Now()

	if req.UpdateDoctor != nil {
		record.DoctorID = *req.UpdateDoctor
	}
	if req.UpdatePatient != nil {
		record.PatientID = *req.UpdatePatient
	}

	if err := s.medicalRecordService.UpdateRecord(record); err != nil {
		return nil, err
	}

	// 📎 2️⃣ Thêm attachments (nếu có)
	for _, attReq := range req.Attachments {
		att := &medicalrecord.Attachment{
			RecordID: record.RecordID,
			FileType: attReq.FileType,
		}

		if _, err := s.attachmentService.AddAttachment(att, attReq.File); err != nil {
			return nil, err
		}
	}

	// 💊 3️⃣ Thêm prescription (nếu có)
	if req.Prescription != nil {
		req.Prescription.MedicalRecordID = &record.RecordID

		// Nếu prescription chưa có PatientID thì gán từ record
		if req.Prescription.PatientID == "" {
			req.Prescription.PatientID = record.PatientID
		}

		if err := s.prescriptionService.CreatePrescription(req.Prescription); err != nil {
			return nil, err
		}
	}

	// ✅ 4️⃣ Update appointment status thành COMPLETED (nếu record có appointment_id)
	if record.AppointmentID != "" {
		if err := s.appointmentService.UpdateStatus(record.AppointmentID, enums.Completed); err != nil {
			// Log error nhưng không fail toàn bộ request
			// Vì medical record đã được update thành công
		}
	}

	// ✅ Trả lại record sau khi hoàn tất
	return s.medicalRecordService.GetRecord(record.RecordID)
}

package doctorrepo

import (
	"hospital-service/internal/models/appointment"
	"hospital-service/internal/models/doctor"
	"time"

	"gorm.io/gorm"
)

type DoctorRepo struct {
	db *gorm.DB
}

func NewDoctorRepo(db *gorm.DB) *DoctorRepo {
	return &DoctorRepo{db: db}
}

// Create thêm doctor mới
func (r *DoctorRepo) Create(d *doctor.Doctor) error {
	return r.db.Create(d).Error
}

// FindByUserID tìm doctor theo UserID từ AuthService
func (r *DoctorRepo) FindByUserID(userID string) (*doctor.Doctor, error) {
	var d doctor.Doctor
	if err := r.db.Where("user_id = ?", userID).First(&d).Error; err != nil {
		return nil, err
	}
	return &d, nil
}

// FindByID tìm doctor theo DoctorID
func (r *DoctorRepo) FindByID(id string) (*doctor.Doctor, error) {
	var d doctor.Doctor
	if err := r.db.First(&d, "doctor_id = ?", id).Error; err != nil {
		return nil, err
	}
	return &d, nil
}

// Update cập nhật thông tin Doctor
func (r *DoctorRepo) Update(d *doctor.Doctor) error {
	return r.db.Save(d).Error
}

// Delete xóa doctor
func (r *DoctorRepo) Delete(id string) error {
	return r.db.Delete(&doctor.Doctor{}, "doctor_id = ?", id).Error
}

// List tất cả doctors
func (r *DoctorRepo) List() ([]doctor.Doctor, error) {
	var doctors []doctor.Doctor
	if err := r.db.Find(&doctors).Error; err != nil {
		return nil, err
	}
	return doctors, nil
}

// FindByHospitalID tìm danh sách bác sĩ theo HospitalID
func (r *DoctorRepo) FindByHospitalID(hospitalID string) ([]doctor.Doctor, error) {
	var doctors []doctor.Doctor
	if err := r.db.Where("hospital_id = ?", hospitalID).Find(&doctors).Error; err != nil {
		return nil, err
	}
	return doctors, nil
}

func (r *DoctorRepo) FindBySlug(slug string) (*doctor.Doctor, error) {
	var d doctor.Doctor
	if err := r.db.Where("slug = ?", slug).First(&d).Error; err != nil {
		return nil, err
	}
	return &d, nil
}
func (r *DoctorRepo) FindBestReplacementDoctor(doctorID string, startTime, endTime time.Time) (*doctor.Doctor, error) {
	var current doctor.Doctor
	if err := r.db.First(&current, "doctor_id = ?", doctorID).Error; err != nil {
		return nil, err
	}

	var candidates []doctor.Doctor
	if err := r.db.Where("specialty = ? AND hospital_id = ? AND doctor_id <> ?",
		current.Specialty, current.HospitalID, doctorID).
		Find(&candidates).Error; err != nil {
		return nil, err
	}

	var bestDoctor *doctor.Doctor
	var bestDiff time.Duration = time.Hour * 24 // cực lớn ban đầu

	for _, d := range candidates {
		var slots []appointment.TimeSlot
		if err := r.db.Where("doctor_id = ? AND DATE(start_time) = ?",
			d.DoctorID, startTime.Format("2006-01-02")).Find(&slots).Error; err != nil {
			continue
		}

		for _, s := range slots {
			if s.AppointmentID != nil && *s.AppointmentID != "" {
				continue // slot đã có bệnh nhân
			}
			diff := absDuration(s.StartTime.Sub(startTime))
			if diff < bestDiff {
				bestDiff = diff
				bestDoctor = &d
			}
		}
	}

	return bestDoctor, nil
}

func absDuration(d time.Duration) time.Duration {
    if d < 0 {
        return -d
    }
    return d
}

package hospitalrepo

import (
	"hospital-service/internal/models/hospital"
	"strings"

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
	err := r.db.
		Preload("Doctors").
		First(&h, "hospital_id = ?", id).Error
	if err != nil {
		return nil, err
	}
	return &h, nil
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
	err := r.db.
		Preload("Doctors").
		Find(&hospitals).Error
	return hospitals, err
}

// ---------------- ListCities ----------------
func (r *HospitalRepo) ListCities() ([]string, error) {
	var cities []string
	err := r.db.Model(&hospital.Hospital{}).
		Distinct().
		Pluck("city", &cities).Error
	return cities, err
}

// ---------------- ListWardsByCity ----------------
func (r *HospitalRepo) ListWardsByCity(city string) ([]string, error) {
	var wards []string
	err := r.db.Model(&hospital.Hospital{}).
		Where("city = ?", city).
		Distinct().
		Pluck("ward", &wards).Error
	return wards, err
}

// ---------------- SearchByAddress ----------------
func (r *HospitalRepo) SearchByAddress(keyword string) ([]hospital.Hospital, error) {
	var hospitals []hospital.Hospital
	if keyword == "" {
		return hospitals, nil
	}

	likeQuery := "%" + strings.ToLower(keyword) + "%"
	err := r.db.
		Where(`
			LOWER(name) LIKE ? OR 
			LOWER(address) LIKE ? OR 
			LOWER(ward) LIKE ? OR 
			LOWER(city) LIKE ?
		`, likeQuery, likeQuery, likeQuery, likeQuery).
		Find(&hospitals).Error

	return hospitals, err
}

// ---------------- ListByCityAndWard ----------------
func (r *HospitalRepo) ListByCityAndWard(city, ward string) ([]hospital.Hospital, error) {
	var hospitals []hospital.Hospital
	query := r.db.Model(&hospital.Hospital{})
	if city != "" {
		query = query.Where("city = ?", city)
	}
	if ward != "" {
		query = query.Where("ward = ?", ward)
	}
	err := query.Find(&hospitals).Error
	return hospitals, err
}

// ---------------- FindNearby ----------------
// Tìm bệnh viện gần tọa độ (lat, lng) trong bán kính (km)
func (r *HospitalRepo) FindNearby(lat, lng, radiusKm float64) ([]hospital.Hospital, error) {
	var hospitals []hospital.Hospital

	// Công thức Haversine (Postgres/MySQL đều hỗ trợ)
	// 6371 là bán kính trái đất (km)
	err := r.db.Raw(`
  SELECT *
  FROM (
    SELECT *, (
      6371 * acos(
        cos(radians(?)) * cos(radians(latitude)) *
        cos(radians(longitude) - radians(?)) +
        sin(radians(?)) * sin(radians(latitude))
      )
    ) AS distance
    FROM hospitals
  ) AS sub
  WHERE distance <= ?
  ORDER BY distance ASC
`, lat, lng, lat, radiusKm).Scan(&hospitals).Error

	return hospitals, err
}

// FindBySlug tìm hospital theo slug
func (r *HospitalRepo) FindBySlug(slug string) (*hospital.Hospital, error) {
	var h hospital.Hospital
	err := r.db.
		Preload("Doctors").
		First(&h, "slug = ?", slug).Error
	if err != nil {
		return nil, err
	}
	return &h, nil
}

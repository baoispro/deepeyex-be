package drugrepo

import (
	"hospital-service/internal/models/drug"
	"strconv"
	"strings"

	"gorm.io/gorm"
)

type DrugRepo struct {
	db *gorm.DB
}

func NewDrugRepo(db *gorm.DB) *DrugRepo {
	return &DrugRepo{db: db}
}

// Create thêm thuốc mới
func (r *DrugRepo) Create(d *drug.Drug) error {
	return r.db.Create(d).Error
}

// GetByID tìm thuốc theo DrugID
func (r *DrugRepo) GetByID(id string) (*drug.Drug, error) {
	var d drug.Drug
	if err := r.db.First(&d, "drug_id = ?", id).Error; err != nil {
		return nil, err
	}
	return &d, nil
}

// ListAll lấy tất cả thuốc
func (r *DrugRepo) ListAll() ([]drug.Drug, error) {
	var ds []drug.Drug
	if err := r.db.Find(&ds).Error; err != nil {
		return nil, err
	}
	return ds, nil
}

// FindWithFilters tìm drugs với filter động
func (r *DrugRepo) FindWithFilters(name, minPrice, maxPrice, minStock, maxStock string) ([]drug.Drug, error) {
	var drugs []drug.Drug
	query := r.db

	// Filter theo name (partial match, case-insensitive)
	if name != "" {
		query = query.Where("LOWER(name) LIKE ?", "%"+strings.ToLower(name)+"%")
	}

	// Filter theo price range
	if minPrice != "" {
		if price, err := strconv.ParseFloat(minPrice, 64); err == nil {
			query = query.Where("price >= ?", price)
		}
	}
	if maxPrice != "" {
		if price, err := strconv.ParseFloat(maxPrice, 64); err == nil {
			query = query.Where("price <= ?", price)
		}
	}

	// Filter theo stock quantity range
	if minStock != "" {
		if stock, err := strconv.Atoi(minStock); err == nil {
			query = query.Where("stock_quantity >= ?", stock)
		}
	}
	if maxStock != "" {
		if stock, err := strconv.Atoi(maxStock); err == nil {
			query = query.Where("stock_quantity <= ?", stock)
		}
	}

	if err := query.Find(&drugs).Error; err != nil {
		return nil, err
	}
	return drugs, nil
}

// Update cập nhật thuốc
func (r *DrugRepo) Update(d *drug.Drug) error {
	return r.db.Save(d).Error
}

// Delete xóa thuốc
func (r *DrugRepo) Delete(id string) error {
	return r.db.Delete(&drug.Drug{}, "drug_id = ?", id).Error
}

func (r *DrugRepo) UpdateStockAndSold(drugID string, quantity int) error {
	return r.db.Model(&drug.Drug{}).
		Where("drug_id = ? AND stock_quantity >= ?", drugID, quantity).
		Updates(map[string]interface{}{
			"stock_quantity": gorm.Expr("stock_quantity - ?", quantity),
			"sold_quantity":  gorm.Expr("sold_quantity + ?", quantity),
		}).Error
}

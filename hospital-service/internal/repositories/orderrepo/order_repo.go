package orderrepo

import (
	"hospital-service/internal/models/order"

	"gorm.io/gorm"
)

type OrderRepo struct {
	db *gorm.DB
}

func NewOrderRepo(db *gorm.DB) *OrderRepo {
	return &OrderRepo{db: db}
}

// ---------------- Create ----------------
func (r *OrderRepo) Create(o *order.Order) error {
	return r.db.Create(o).Error
}

// ---------------- GetByID ----------------
func (r *OrderRepo) GetByID(id string) (*order.Order, error) {
	var o order.Order
	if err := r.db.Preload("OrderItems.Drug").Preload("Patient").First(&o, "order_id = ?", id).Error; err != nil {
		return nil, err
	}
	return &o, nil
}

// -------------- GetByPatientID --------------
func (r *OrderRepo) FindByPatientID(patientID string) ([]order.Order, error) {
	var orders []order.Order
	if err := r.db.Where("patient_id = ?", patientID).Preload("OrderItem.Drug").Preload("Patient").Find(&orders).Error; err != nil {
		return nil, err
	}
	return orders, nil
}


// ---------------- ListAll ----------------
func (r *OrderRepo) ListAll() ([]order.Order, error) {
	var orders []order.Order
	if err := r.db.Preload("OrderItems.Drug").Preload("Patient").Find(&orders).Error; err != nil {
		return nil, err
	}
	return orders, nil
}

// ---------------- Update ----------------
func (r *OrderRepo) Update(o *order.Order) error {
	return r.db.Save(o).Error
}

// ---------------- Delete ----------------
func (r *OrderRepo) Delete(id string) error {
	return r.db.Delete(&order.Order{}, "order_id = ?", id).Error
}


func (r *OrderRepo) BeginTx() *gorm.DB {
	return r.db.Begin()
}
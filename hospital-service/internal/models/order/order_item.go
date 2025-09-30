package order

import "hospital-service/internal/models/drug"

type OrderItem struct {
	OrderItemID string    `gorm:"column:order_item_id;primaryKey;size:36" json:"order_item_id"`
	OrderID     string    `gorm:"column:order_id;size:36;not null" json:"order_id"`
	DrugID      string    `gorm:"column:drug_id;size:36" json:"drug_id"`
	DrugName    string    `gorm:"column:drug_name;size:255" json:"drug_name"`
	Service     string    `gorm:"column:service;size:255;" json:"service"`
	Quantity    int       `gorm:"not null;default:1" json:"quantity"`
	Price       float64   `gorm:"type:decimal(10,2);not null" json:"price"`
	Drug        *drug.Drug `gorm:"foreignKey:DrugID;references:DrugID" json:"drug,omitempty"`
}

package drug

import "time"

type Drug struct {
    DrugID          uint64    `gorm:"column:drug_id;primaryKey;autoIncrement" json:"drug_id"`
    Name            string    `gorm:"size:100;not null" json:"name"`
    Description     string    `gorm:"type:text" json:"description"`
    Price           float64   `gorm:"type:decimal(10,2);not null" json:"price"`
    StockQuantity   int       `gorm:"not null;default:0" json:"stock_quantity"`
    DiscountPercent float64   `gorm:"type:decimal(5,2);default:0" json:"discount_percent"`
    CreatedAt       time.Time `gorm:"autoCreateTime" json:"created_at"`
    UpdatedAt       time.Time `gorm:"autoUpdateTime" json:"updated_at"`
}

func (Drug) TableName() string {
    return "drugs"
}

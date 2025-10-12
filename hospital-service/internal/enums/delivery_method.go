package enums

type DeliveryMethod string

const (
	// PICKUP nhận tại bệnh viện/phòng khám
	PICKUP DeliveryMethod = "PICKUP"
	
	// HOME_DELIVERY giao hàng tận nơi
	HOME_DELIVERY DeliveryMethod = "HOME_DELIVERY"
	
	// EXPRESS_DELIVERY giao hàng nhanh
	EXPRESS_DELIVERY DeliveryMethod = "EXPRESS_DELIVERY"
)

// IsValid kiểm tra delivery method có hợp lệ không
func (d DeliveryMethod) IsValid() bool {
	switch d {
	case PICKUP, HOME_DELIVERY, EXPRESS_DELIVERY:
		return true
	}
	return false
}


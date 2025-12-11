package statisticservice

import (
	"hospital-service/internal/enums"
	"hospital-service/internal/repositories/appointmentrepo"
	"hospital-service/internal/repositories/orderrepo"
	"time"

	"gorm.io/gorm"
)

type StatisticService struct {
	db              *gorm.DB
	orderRepo       *orderrepo.OrderRepo
	appointmentRepo *appointmentrepo.AppointmentRepo
}

func NewStatisticService(db *gorm.DB, orderRepo *orderrepo.OrderRepo, appointmentRepo *appointmentrepo.AppointmentRepo) *StatisticService {
	return &StatisticService{
		db:              db,
		orderRepo:       orderRepo,
		appointmentRepo: appointmentRepo,
	}
}

// StatisticResponse response cho thống kê tổng quan
type StatisticResponse struct {
	TotalOrders           int64                          `json:"total_orders"`            // Tổng số đơn hàng
	TotalRevenue          float64                        `json:"total_revenue"`           // Tổng doanh thu
	TotalBookings         int64                          `json:"total_bookings"`          // Tổng số đặt lịch
	CompletedAppointments int64                          `json:"completed_appointments"` // Số lượng hoàn thành khám bệnh
	AppointmentTimeline   []AppointmentTimelineItem      `json:"appointment_timeline"`   // Timeline lưu lượng đặt lịch
	OrderStatusStats      []OrderStatusStat              `json:"order_status_stats"`     // Thống kê trạng thái đơn
	RevenueByService      []RevenueByServiceItem         `json:"revenue_by_service"`    // Doanh thu theo dịch vụ
}

// AppointmentTimelineItem item trong timeline đặt lịch
type AppointmentTimelineItem struct {
	Date  string `json:"date"`  // Format: YYYY-MM-DD
	Count int64  `json:"count"` // Số lượng đặt lịch trong ngày
}

// OrderStatusStat thống kê theo trạng thái đơn
type OrderStatusStat struct {
	Status string `json:"status"` // PENDING, PAID, CANCELED, DELIVERED
	Count  int64  `json:"count"`  // Số lượng
}

// RevenueByServiceItem doanh thu theo dịch vụ
type RevenueByServiceItem struct {
	ServiceID   string  `json:"service_id"`
	ServiceName string  `json:"service_name"`
	Revenue     float64 `json:"revenue"` // Tổng doanh thu
	OrderCount  int64   `json:"order_count"` // Số lượng đơn
}

// GetStatistics lấy thống kê tổng quan
// startDate và endDate là optional, nếu không có thì lấy tất cả
func (s *StatisticService) GetStatistics(startDate, endDate *time.Time) (*StatisticResponse, error) {
	response := &StatisticResponse{}

	// 1. Tổng số đơn hàng
	var totalOrders int64
	orderQuery := s.db.Table("orders")
	if startDate != nil {
		orderQuery = orderQuery.Where("created_at >= ?", *startDate)
	}
	if endDate != nil {
		endDateInclusive := endDate.Add(24 * time.Hour).Add(-1 * time.Second)
		orderQuery = orderQuery.Where("created_at <= ?", endDateInclusive)
	}
	if err := orderQuery.Count(&totalOrders).Error; err != nil {
		return nil, err
	}
	response.TotalOrders = totalOrders

	// 2. Tổng doanh thu (chỉ tính orders có status PAID hoặc DELIVERED)
	var totalRevenue float64
	revenueQuery := s.db.Model(&struct {
		TotalAmount float64 `gorm:"column:total_amount"`
	}{}).Table("orders").
		Where("status IN ?", []string{string(enums.PAID), string(enums.DELIVERED)})
	if startDate != nil {
		revenueQuery = revenueQuery.Where("created_at >= ?", *startDate)
	}
	if endDate != nil {
		endDateInclusive := endDate.Add(24 * time.Hour).Add(-1 * time.Second)
		revenueQuery = revenueQuery.Where("created_at <= ?", endDateInclusive)
	}
	if err := revenueQuery.Select("COALESCE(SUM(total_amount), 0)").Scan(&totalRevenue).Error; err != nil {
		return nil, err
	}
	response.TotalRevenue = totalRevenue

	// 3. Tổng số đặt lịch (appointments)
	var totalBookings int64
	appointmentQuery := s.db.Table("appointments")
	if startDate != nil {
		appointmentQuery = appointmentQuery.Where("created_at >= ?", *startDate)
	}
	if endDate != nil {
		endDateInclusive := endDate.Add(24 * time.Hour).Add(-1 * time.Second)
		appointmentQuery = appointmentQuery.Where("created_at <= ?", endDateInclusive)
	}
	if err := appointmentQuery.Count(&totalBookings).Error; err != nil {
		return nil, err
	}
	response.TotalBookings = totalBookings

	// 4. Số lượng hoàn thành khám bệnh
	var completedAppointments int64
	completedQuery := s.db.Model(&struct {
		AppointmentID string `gorm:"column:appointment_id"`
	}{}).Table("appointments").
		Where("status IN ?", []string{string(enums.Completed), string(enums.CompletedOnline)})
	if startDate != nil {
		completedQuery = completedQuery.Where("created_at >= ?", *startDate)
	}
	if endDate != nil {
		endDateInclusive := endDate.Add(24 * time.Hour).Add(-1 * time.Second)
		completedQuery = completedQuery.Where("created_at <= ?", endDateInclusive)
	}
	if err := completedQuery.Count(&completedAppointments).Error; err != nil {
		return nil, err
	}
	response.CompletedAppointments = completedAppointments

	// 5. Timeline lưu lượng đặt lịch (theo ngày)
	timeline, err := s.getAppointmentTimeline(startDate, endDate)
	if err != nil {
		return nil, err
	}
	response.AppointmentTimeline = timeline

	// 6. Thống kê trạng thái đơn
	orderStatusStats, err := s.getOrderStatusStats(startDate, endDate)
	if err != nil {
		return nil, err
	}
	response.OrderStatusStats = orderStatusStats

	// 7. Doanh thu theo dịch vụ (từ order_items có service_id, join với orders có appointment_id)
	revenueByService, err := s.getRevenueByService(startDate, endDate)
	if err != nil {
		return nil, err
	}
	response.RevenueByService = revenueByService

	return response, nil
}

// getAppointmentTimeline lấy timeline đặt lịch theo ngày
func (s *StatisticService) getAppointmentTimeline(startDate, endDate *time.Time) ([]AppointmentTimelineItem, error) {
	var results []struct {
		Date  string `gorm:"column:date"`
		Count int64  `gorm:"column:count"`
	}

	query := "SELECT DATE(created_at) as date, COUNT(*) as count FROM appointments WHERE 1=1"
	var args []interface{}

	if startDate != nil {
		query += " AND created_at >= ?"
		args = append(args, *startDate)
	}
	if endDate != nil {
		endDateInclusive := endDate.Add(24 * time.Hour).Add(-1 * time.Second)
		query += " AND created_at <= ?"
		args = append(args, endDateInclusive)
	}

	query += " GROUP BY DATE(created_at) ORDER BY date ASC"

	if err := s.db.Raw(query, args...).Scan(&results).Error; err != nil {
		return nil, err
	}

	timeline := make([]AppointmentTimelineItem, len(results))
	for i, r := range results {
		timeline[i] = AppointmentTimelineItem{
			Date:  r.Date,
			Count: r.Count,
		}
	}

	return timeline, nil
}

// getOrderStatusStats lấy thống kê trạng thái đơn
func (s *StatisticService) getOrderStatusStats(startDate, endDate *time.Time) ([]OrderStatusStat, error) {
	var results []struct {
		Status string `gorm:"column:status"`
		Count  int64  `gorm:"column:count"`
	}

	query := "SELECT status, COUNT(*) as count FROM orders WHERE 1=1"
	var args []interface{}

	if startDate != nil {
		query += " AND created_at >= ?"
		args = append(args, *startDate)
	}
	if endDate != nil {
		endDateInclusive := endDate.Add(24 * time.Hour).Add(-1 * time.Second)
		query += " AND created_at <= ?"
		args = append(args, endDateInclusive)
	}

	query += " GROUP BY status ORDER BY status ASC"

	if err := s.db.Raw(query, args...).Scan(&results).Error; err != nil {
		return nil, err
	}

	stats := make([]OrderStatusStat, len(results))
	for i, r := range results {
		stats[i] = OrderStatusStat{
			Status: r.Status,
			Count:  r.Count,
		}
	}

	return stats, nil
}

// getRevenueByService lấy doanh thu theo dịch vụ (từ order_items có service_id, join với orders có appointment_id)
func (s *StatisticService) getRevenueByService(startDate, endDate *time.Time) ([]RevenueByServiceItem, error) {
	var results []struct {
		ServiceID   string  `gorm:"column:service_id"`
		ServiceName string  `gorm:"column:service_name"`
		Revenue     float64 `gorm:"column:revenue"`
		OrderCount  int64   `gorm:"column:order_count"`
	}

	query := `
		SELECT 
			oi.service_id,
			COALESCE(s.name, oi.item_name) as service_name,
			COALESCE(SUM(oi.price * oi.quantity), 0) as revenue,
			COUNT(DISTINCT o.order_id) as order_count
		FROM order_items oi
		INNER JOIN orders o ON oi.order_id = o.order_id
		LEFT JOIN services s ON oi.service_id = s.service_id
		WHERE oi.service_id IS NOT NULL 
			AND oi.service_id != ''
			AND o.appointment_id IS NOT NULL
			AND o.appointment_id != ''
			AND o.status IN (?, ?)
	`

	args := []interface{}{string(enums.PAID), string(enums.DELIVERED)}

	if startDate != nil {
		query += " AND o.created_at >= ?"
		args = append(args, *startDate)
	}
	if endDate != nil {
		endDateInclusive := endDate.Add(24 * time.Hour).Add(-1 * time.Second)
		query += " AND o.created_at <= ?"
		args = append(args, endDateInclusive)
	}

	query += " GROUP BY oi.service_id, s.name, oi.item_name ORDER BY revenue DESC"

	if err := s.db.Raw(query, args...).Scan(&results).Error; err != nil {
		return nil, err
	}

	revenueByService := make([]RevenueByServiceItem, len(results))
	for i, r := range results {
		revenueByService[i] = RevenueByServiceItem{
			ServiceID:   r.ServiceID,
			ServiceName: r.ServiceName,
			Revenue:     r.Revenue,
			OrderCount:  r.OrderCount,
		}
	}

	return revenueByService, nil
}


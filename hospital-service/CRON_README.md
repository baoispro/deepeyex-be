# Cron Job - Tự động tạo Time Slots

## Mô tả
Hệ thống cron job tự động tạo time slots cho tất cả bác sĩ vào 19h thứ 7 hàng tuần.

## Lịch trình
- **Thời gian chạy**: 19:00 thứ 7 hàng tuần (Asia/Ho_Chi_Minh timezone)
- **Cron expression**: `0 19 * * 6`

## Time Slots được tạo
Mỗi bác sĩ sẽ có các time slots sau cho mỗi ngày trong tuần:

### Ca sáng
- 8:30 - 9:00
- 9:00 - 10:00  
- 10:00 - 11:00
- 11:00 - 12:00

### Ca chiều
- 13:30 - 14:00
- 14:00 - 15:00
- 15:00 - 16:00
- 16:00 - 17:00

### Ca tối
- 18:00 - 19:00
- 19:00 - 20:00
- 20:00 - 21:00

## Cách hoạt động
1. Vào 19h thứ 7, cron job sẽ tự động chạy
2. Lấy danh sách tất cả bác sĩ từ database
3. Tạo time slots cho tuần tiếp theo (từ thứ 2 đến chủ nhật)
4. Xóa time slots cũ nếu có (để tránh trùng lặp)
5. Tạo time slots mới với capacity = 1 cho mỗi slot

## Cấu trúc file
- `internal/services/cronservice/cron_service.go`: Service chính cho cron job
- `internal/services/timeslotservice/time_slot_service.go`: Service tạo time slots
- `internal/repositories/appointmentrepo/time_slot_repo.go`: Repository cho time slots

## Test
Để test cron job, có một job test chạy mỗi phút (có thể xóa trong production):
```go
// Trong cron_service.go
_, err = s.cron.AddFunc("* * * * *", s.testJob)
```

## Logs
Cron job sẽ ghi log khi:
- Khởi động cron service
- Chạy job tạo time slots
- Lỗi xảy ra trong quá trình tạo time slots
- Thời gian thực hiện job

## Dependencies
- `github.com/robfig/cron/v3`: Library cron job
- `timezone`: Asia/Ho_Chi_Minh

## Lưu ý
- Cron job chỉ tạo time slots cho tuần tiếp theo
- Mỗi slot có capacity = 1 (1 appointment per slot)
- Time slots cũ sẽ bị xóa trước khi tạo mới để tránh trùng lặp
- Hệ thống sử dụng timezone Việt Nam

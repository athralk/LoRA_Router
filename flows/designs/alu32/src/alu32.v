// 32-bit ALU with registered inputs and output: add, sub, and, or, xor, shifts, compare.
module alu32 (
    input  wire        clk,
    input  wire        rst,
    input  wire [3:0]  op,
    input  wire [31:0] a,
    input  wire [31:0] b,
    output reg  [31:0] y,
    output reg         zero
);
    reg [3:0]  rop;
    reg [31:0] ra, rb;
    reg [31:0] r;
    always @(*) begin
        case (rop)
            4'd0: r = ra + rb;
            4'd1: r = ra - rb;
            4'd2: r = ra & rb;
            4'd3: r = ra | rb;
            4'd4: r = ra ^ rb;
            4'd5: r = ra << rb[4:0];
            4'd6: r = ra >> rb[4:0];
            4'd7: r = $signed(ra) >>> rb[4:0];
            4'd8: r = {31'd0, $signed(ra) < $signed(rb)};
            4'd9: r = {31'd0, ra < rb};
            default: r = ~(ra | rb);
        endcase
    end
    always @(posedge clk) begin
        if (rst) begin
            rop <= 0; ra <= 0; rb <= 0; y <= 0; zero <= 0;
        end else begin
            rop <= op; ra <= a; rb <= b; y <= r; zero <= (r == 32'd0);
        end
    end
endmodule

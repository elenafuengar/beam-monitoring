class SPSOctupolesNew(object):

    def __init__(self, optics='Q20'):
        if optics == 'Q20':
            self.coeffs = {
                'daxx_f': 800.5951703666668,
                'daxy_f': -561.4469519999999,
                'dayy_f': 98.46045846,
                'daxx_d': 96.25418287999999,
                'daxy_d': -543.9761326666667,
                'dayy_d': 760.1132440666667,
                'd_q2x_f': -434.33259977333336,
                'd_q2y_f': 153.72139112,
                'd_q2x_d': 37.35988630666667,
                'd_q2y_d': -749.8626844666666}
        elif optics == 'Q26':
            self.coeffs = {
                'daxx_f': 771.4425583466667,
                'daxy_f': -370.0765931066667,
                'dayy_f': 44.40283461333333,
                'daxx_d': 42.637013153333335,
                'daxy_d': -354.80800317333336,
                'dayy_d': 738.4474973066666,
                'd_q2x_f': 403.1826211333334,
                'd_q2y_f': -96.43585381999999,
                'd_q2x_d': 44.411458933333314,
                'd_q2y_d': -184.71047933333332}
        else:
            raise ValueError("Optics %s unknown! Use either 'Q20' or 'Q26'."%
                             optics)

    def get_anharmonicities(self, KLOF, KLOD):
        axx = self.coeffs['daxx_f'] * KLOF + self.coeffs['daxx_d'] * KLOD
        axy = self.coeffs['daxy_f'] * KLOF + self.coeffs['daxy_d'] * KLOD
        ayy = self.coeffs['dayy_f'] * KLOF + self.coeffs['dayy_d'] * KLOD

        return axx, axy, ayy

    def get_q2(self, KLOF, KLOD):
        q2x = self.coeffs['d_q2x_f'] * KLOF + self.coeffs['d_q2x_d'] * KLOD
        q2y = self.coeffs['d_q2y_f'] * KLOF + self.coeffs['d_q2y_d'] * KLOD

        return q2x, q2y

    def get_q1_feeddown(self, KLOF, KLOD, dp_offset=-1.7e-4):
        q2x, q2y = self.get_q2(KLOF, KLOD)
        q1x_fd = q2x * dp_offset
        q1y_fd = q2y * dp_offset

        return q1x_fd, q1y_fd

    def get_q0_feeddown(self, KLOF, KLOD, dp_offset=-1.7e-4):
        q2x, q2y = self.get_q2(KLOF, KLOD)
        q0x_fd = q2x / 2 * dp_offset * dp_offset
        q0y_fd = q2y / 2 * dp_offset * dp_offset

        return q0x_fd, q0y_fd

    def apply_to_machine(self, machine, KLOF, KLOD, dp_offset):
            q1x_fd, q1y_fd = self.get_q1_feeddown(KLOF, KLOD, dp_offset)
            machine.Qp_x[0] += q1x_fd
            machine.Qp_y[0] += q1y_fd

            q2x, q2y = self.get_q2(KLOF, KLOD)
            try:
                machine.Qp_x[1] += q2x
                machine.Qp_y[1] += q2y
            except IndexError:
                machine.Qp_x += [ q2x ]
                machine.Qp_y += [ q2y ]

            axx, axy, ayy = self.get_anharmonicities(KLOF, KLOD)
            machine.app_x += axx
            machine.app_y += ayy
            machine.app_xy += axy
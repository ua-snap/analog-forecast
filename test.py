import numpy as np
import xarray as xr
import analog_forecast as af


def test_take_analogs():
    # set up a DataArray that should yield times 2020-01-01 12:00:00 and 1975-02-22 12:00:00
    # for take_analogs called with buffer=30 and n=2
    test_times = np.array(
        [
            d + " 12:00:00" 
            for d in ["2020-01-01", "2020-01-29", "2019-12-17", "1975-02-03", "1975-02-22"]
        ], 
        dtype=np.datetime64
    )
    test_da = xr.DataArray(
        data=[1, 2, 3, 4, 1.2],
        dims=["time"],
        coords=dict(
            time=test_times,
        ),
    )
    test = af.take_analogs(test_da, 30, 2)
    assert np.all(test == test_da.sel(time=test_times[[0, -1]]))
    
    
if __name__ == "__main__":
    test_take_analogs()
    print("all tests passed")
